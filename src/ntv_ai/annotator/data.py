"""Freeze human-reviewed sources without modifying the reviewer or original images."""
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import zipfile

from PIL import Image, ImageOps

from ntv_ai.annotator.splits import fingerprints, grouped_split
from ntv_ai.data.views import eligibility
from ntv_ai.domain.records import Annotation, Geometry, ImageRecord
from ntv_ai.experiments.config import read_config
from ntv_ai.storage import atomic_json, digest, file_hash, inside, read_snapshot, safe_output, write_snapshot


CLASSES = ["torre", "isolador", "fundacao_concreto", "fundacao_estai", "espacador", "amortecedor", "sinalizacao"]


def read_reviews(path, coco_hash=None):
    """One read transaction, including WAL, without writing to the live database."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        db.execute("BEGIN")
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        if coco_hash and metadata.get("source_sha256") != coco_hash:
            raise ValueError("Review database does not belong to this proposal COCO")
        rows = [{**json.loads(s), "revision": rev} for rev, s in db.execute("SELECT revision,state FROM reviews ORDER BY image_id")]
        last_event = db.execute("SELECT MAX(id) FROM events").fetchone()[0]
        has_actors = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actors'").fetchone()
        actors = [{"id": row[0], "name": row[1]} for row in db.execute("SELECT id,name FROM actors ORDER BY id")] if has_actors else []
    return {"metadata": metadata, "last_event": last_event, "states": rows, "actors": actors}


def protected_hashes(coco, review):
    images = {im["id"]: im for im in coco["images"]}
    return {images[s["image_id"]]["sha256"] for s in review["states"]
            if s["status"] != "pending" or s.get("revision", 0) > 0 or s.get("review_stage") in ("reviewed", "validated")}


def authoritative_union(candidates):
    """Merge byte OR pixel duplicates; latest approved source replaces all boxes."""
    parents = list(range(len(candidates)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, (r, _) in enumerate(candidates):
        for kind, value in (("bytes", r.sha256), ("pixels", r.pixel_sha256)):
            if value:
                key = kind, value
                if key in seen:
                    parents[find(i)] = find(seen[key])
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(candidates):
        groups[find(i)].append(row)
    records, replaced = [], []
    for members in groups.values():
        if len({r.metadata["capture_id"] for r, _ in members}) != 1:
            raise ValueError("Exact duplicate across captures needs adjudication")
        members.sort(key=lambda row: (row[1], row[0].image_id))
        r = deepcopy(members[-1][0])
        r.metadata["source_versions"] = [{"image_id": m.image_id, "priority": priority,
                                          "boxes": len(m.annotations)} for m, priority in members]
        r.image_id = digest(["ntv-human-development", r.sha256])
        # Annotation IDs must be unique even across exports with repeated integer IDs.
        for a in r.annotations:
            a.annotation_id = digest([r.image_id, a.annotation_id])
        records.append(r)
        if len(members) > 1:
            replaced.append({"sha256": r.sha256, "kept_priority": members[-1][1],
                             "members": r.metadata["source_versions"]})
    return sorted(records, key=lambda r: r.image_id), replaced


def freeze(config_path):
    cfg = read_config(config_path)
    out = Path(cfg["output"]).resolve()
    baseline_path = Path(cfg["baseline_manifest"])
    baseline, old = read_snapshot(baseline_path)
    source_path, db_path = Path(cfg["review_coco"]), Path(cfg["review_database"])
    image_root = Path(cfg["image_root"]).resolve()
    inventory_path = Path(cfg["inventory"])
    protected = [image_root, baseline_path.parent, source_path.parent, db_path.parent]
    protected += [Path(s["root"]) for s in baseline["sources"].values()]
    safe_output(out, protected)
    if out.exists():
        raise FileExistsError("Use a new output for every dataset version")
    coco = json.loads(source_path.read_text(encoding="utf-8"))
    if [c["name"] for c in coco["categories"]] != CLASSES:
        raise ValueError("Unexpected taxonomy/order")
    coco_hash = file_hash(source_path)
    review = read_reviews(db_path, coco_hash)
    if cfg.get("calibration_archive"):
        archive = Path(cfg["calibration_archive"])
        if file_hash(archive) != cfg["calibration_sha256"]:
            raise ValueError("Calibration archive changed")
        with zipfile.ZipFile(archive) as z:
            calibration = json.loads(z.read("aceitas.coco.json"))
        if len(calibration["images"]) != 30 or len(calibration["annotations"]) != 618:
            raise ValueError("Expected corrected calibration (30 images / 618 boxes)")
        if baseline["lineage"].get("input_zip_sha256") != cfg["calibration_sha256"]:
            raise ValueError("Baseline has not incorporated the corrected calibration")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory = [r for r in inventory if r.get("kind") == "image"]
    by_hash = defaultdict(list)
    anchors = {}
    for row in inventory:
        by_hash[row["source_sha256"]].append(row)
        capture = row["capture"]
        if row.get("datetime"):
            anchors[capture] = min(anchors.get(capture, row["datetime"]), row["datetime"])
    def enrich(r):
        matches = by_hash[r.sha256]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or absent inventory match: {r.sha256}")
        row = matches[0]
        path = inside(image_root, row["copy_relative_path"])
        if file_hash(path) != r.sha256:
            raise ValueError(f"Image hash mismatch: {path}")
        with Image.open(path) as opened:
            if opened.size != (r.width, r.height):
                raise ValueError("Image size mismatch")
            rgb = opened.convert("RGB")
            pixels = hashlib.sha256(str(rgb.size).encode()); pixels.update(rgb.tobytes())
            r.pixel_sha256 = pixels.hexdigest()
            # Fixed full-resolution decode, then deterministic thumbnails.
            fp = fingerprints(rgb)
        r.metadata.update(capture_id=row["capture"], datetime=row["datetime"], **fp)
        r.metadata["historical_split"] = r.split
        r.split = None
        r.dataset_id = "ntv_human"
        r.path = row["copy_relative_path"]
        r.annotation_scope = dict.fromkeys(CLASSES, "exhaustive")
        return r
    candidates = []
    for r in old:
        if r.metadata.get("transform") or r.dataset_id.startswith("tiles:"):
            continue
        r.metadata.setdefault("review_stage", "reviewed")
        r.metadata["validation_evidence"] = r.metadata.get("coverage_confirmation")
        candidates.append((enrich(r), 0))
    images = {im["id"]: im for im in coco["images"]}
    category = {c["id"]: c["name"] for c in coco["categories"]}
    for state in review["states"]:
        if state["status"] != "accepted":
            continue
        if state.get("reviewed_resolution") != "original":
            raise ValueError("Accepted preview needs original-resolution review before training")
        im = images[state["image_id"]]
        annotations = []
        for box in state["boxes"]:
            x, y, w, h = box["bbox"]
            label = category[box["category_id"]]
            annotations.append(Annotation(str(box["id"]), label, Geometry("bbox", [x, y, x+w, y+h]),
                {"review_coco_sha256": coco_hash, "review_image_id": state["image_id"],
                 "review_revision": state["revision"]}, label=label, mapping_status="mapped",
                attributes={"proposal_score": box.get("score"), "review_stage": state.get("review_stage", "reviewed")},
                original=deepcopy(box)))
        r = ImageRecord(str(im["id"]), "ntv_human", im["file_name"], im["width"], im["height"],
                        im["sha256"], annotations=annotations,
                        metadata={"review_stage": state.get("review_stage", "reviewed"),
                                  "review_evidence": deepcopy(state), "annotation_origin": "review_database"})
        candidates.append((enrich(r), 1))
    export_hashes = {}
    for priority, export_path in enumerate(cfg.get("human_exports", []), start=2):
        path = Path(export_path)
        export_hashes[str(path)] = file_hash(path)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                document = json.loads(archive.read("aceitas.coco.json"))
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
        labels = {c["id"]: c["name"] for c in document["categories"]}
        if set(labels.values()) != set(CLASSES):
            raise ValueError("Human export taxonomy mismatch")
        export_annotations = defaultdict(list)
        for ann in document["annotations"]:
            export_annotations[ann["image_id"]].append(ann)
        for im in document["images"]:
            evidence = im.get("review_evidence", {})
            stage = im.get("review_stage")
            if stage not in ("reviewed", "validated") or evidence.get("status") != "accepted" or evidence.get("reviewed_resolution") != "original":
                raise ValueError("Only explicit original-resolution human exports may override references")
            annotations = []
            for ann in export_annotations[im["id"]]:
                x, y, w, h = ann["bbox"]; label = labels[ann["category_id"]]
                box_id = ann.get("attributes", {}).get("review_box_id")
                if not box_id:
                    raise ValueError("Human export lost review box identifiers")
                annotations.append(Annotation(box_id, label, Geometry("bbox", [x, y, x+w, y+h]),
                    {"human_export_sha256": export_hashes[str(path)]}, label=label, mapping_status="mapped",
                    attributes=ann.get("attributes", {}), original={**ann, "id": box_id}))
            r = ImageRecord(str(im["id"]), "ntv_human", im["file_name"], im["width"], im["height"], im["sha256"],
                            annotations=annotations, metadata={"review_stage": stage, "review_evidence": evidence,
                                                               "annotation_origin": "human_export"})
            candidates.append((enrich(r), priority))
    records, duplicates = authoritative_union(candidates)
    options = cfg["split"]
    previous = None
    if cfg.get("previous_partition"):
        previous_path = Path(cfg["previous_partition"])
        previous = json.loads(previous_path.read_text())
        if "group_assignment" not in previous:
            _, previous_records = read_snapshot(previous_path.parent / "full640/manifest.json")
            previous["group_assignment"] = {r.group_id: r.split or "buffer" for r in previous_records}
    partitions = grouped_split(records, anchors, CLASSES, options, previous)
    sources = {"ntv_human": {"root": str(image_root), "version": digest({r.path: r.sha256 for r in records}),
                              "config": {"classes": CLASSES}, "evidence": {}}}
    active = [r for r in records if r.split in ("train", "val")]
    problems = eligibility(active, CLASSES)
    if problems:
        raise ValueError(f"Invalid reviewed data: {problems[:10]}")
    lineage = {"operation": "annotator_development", "baseline_version": baseline["version"],
               "reviews_sha256": digest(review), "inventory_sha256": file_hash(inventory_path),
               "recipe_sha256": digest(cfg), "human_exports": export_hashes,
               "frozen_at": datetime.now(timezone.utc).isoformat()}
    view = {"id": "annotator-full640", "classes": CLASSES, "purpose": "exploratory", "eligible": True,
            "image_ids": [r.image_id for r in active], "evaluation_image_ids": [r.image_id for r in active],
            "splits": {"mode": "grouped_development", "test": None}, "tiling": {"enabled": False}}
    manifest = write_snapshot(out / "full640", records, sources, lineage, [], view=view)
    atomic_json(out / "partition.json", partitions)
    atomic_json(out / "reviews-frozen.json", review)
    atomic_json(out / "duplicates.json", duplicates)
    atomic_json(out / "sources.json", {"config": cfg, "lineage": lineage, "anchors": anchors,
                                      "protected_hashes": sorted(protected_hashes(coco, review) | {r.sha256 for r in records})})
    contact_sheet(out, records, image_root, partitions["pairs"])
    return {"manifest": str(manifest), "output": str(out), "support": partitions["support"],
            "warnings": partitions["warnings"]}


def contact_sheet(output, records, root, pairs):
    """Read-only source images; small, explicitly identified evidence thumbnails."""
    folder = output / "similarity"; folder.mkdir()
    lookup = {r.image_id: r for r in records}
    ids = {p[k] for p in pairs for k in ("a", "b")}
    for key in ids:
        with Image.open(inside(root, lookup[key].path)) as image:
            ImageOps.contain(image.convert("RGB"), (360, 240)).save(folder / (key + ".jpg"), quality=85)
    body = ["<!doctype html><meta charset='utf-8'><title>NTV: temporal and visual pairs</title>",
            "<style>body{font:14px Arial}article{display:flex;gap:12px;margin:20px}img{width:360px;height:240px;object-fit:contain}figure{margin:0}</style>",
            "<h1>Grouping evidence</h1><p>Similarity is a heuristic, not proof of independence.</p>"]
    for pair in pairs:
        panels = []
        for key in ("a", "b"):
            r = lookup[pair[key]]
            panels.append(f'<figure><img src="{r.image_id}.jpg"><figcaption>{html.escape(Path(r.path).name)} — {r.split or "buffer"}</figcaption></figure>')
        body.append("<article>" + "".join(panels) + "</article>")
    (folder / "index.html").write_text("\n".join(body), encoding="utf-8")
