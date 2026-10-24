"""Immutable preparation of the reviewed filtered subsets and original NTV lot."""
import csv
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ntv_ai.data.adapters.readers import SourceReader, CocoReader, LabelMeReader, YoloReader
from ntv_ai.data.groups import group_records
from ntv_ai.data.harmonize import harmonize
from ntv_ai.data.stratified import stratify_groups
from ntv_ai.data.views import eligibility
from ntv_ai.domain.records import Geometry
from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.runs import Run
from ntv_ai.pipelines.benchmark_data import descriptor, temporal_pilot, tile_impact
from ntv_ai.pipelines.prepare import materialize_tiles
from ntv_ai.storage import atomic_json, digest, file_hash, write_snapshot


CLASSES = {
    "ptl_ai": ["isolador", "espacador", "amortecedor", "sinalizacao"],
    "mpid": ["isolador"],
    "stn_plad": ["torre", "isolador", "espacador", "amortecedor", "sinalizacao"],
    "ttpla": ["torre"],
    "ntv_ai": ["torre", "isolador", "fundacao_concreto", "fundacao_estai", "espacador", "amortecedor", "sinalizacao"],
}
FOLDERS = {"ptl_ai": "FURNAS", "mpid": "MPID", "stn_plad": "PLAD", "ttpla": "TTPLA"}


def read_filtered_yolo(name, root, workers):
    reader = SourceReader({"id": name, "images": ["**/images/*"], "workers": workers}, root)
    reader.inventory()
    if name == "ptl_ai":
        names_path = root / "labels.txt"
        reader.remember(names_path)
        names = names_path.read_text(encoding="utf-8-sig").splitlines()
    for r in reader.records:
        image = root / r.path
        labels = image.parent.parent / "labels" / (image.stem + ".txt")
        reader.remember(labels)
        if name == "mpid":
            material = {"glass": "vidro", "porcelain": "ceramica", "composite": "polimero"}[Path(r.path).parts[0]]
            r.metadata["material"] = material
            names = ["isolador"]
        for i, line in enumerate(labels.read_text().splitlines()):
            if not line.strip():
                continue
            c, x, y, w, h = map(float, line.split())
            if not all(math.isfinite(v) for v in (c, x, y, w, h)) or c != int(c) or not 0 <= c < len(names):
                raise ValueError(f"Invalid YOLO annotation: {labels}:{i + 1}")
            box = [(x-w/2)*r.width, (y-h/2)*r.height, (x+w/2)*r.width, (y+h/2)*r.height]
            reader.add(r, labels, i, names[int(c)], Geometry("bbox", box), {"line": line, "class_id": int(c)})
    for p in root.rglob("*.csv"):
        reader.remember(p)
    for p in root.rglob("classes.txt"):
        reader.remember(p)
    return reader, reader.finish()


def evidence_groups(records, metadata):
    """Rebind historical evidence by image hash, not paths changed by filtering."""
    original, parents = {}, {}
    inventories = sorted(metadata.glob("*_images.csv"))
    for p in inventories:
        with p.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                original[(row["dataset"], row["image_key"])] = row["sha256"]

    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(a, b):
        parents[find(a)] = find(b)

    review = metadata / "similarity_review.csv"
    with review.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["review_status"] in ("visually_equivalent", "visually_correlated_views"):
                a = original.get((row["dataset_a"], row["image_a"]))
                b = original.get((row["dataset_b"], row["image_b"]))
                if a and b:
                    union(a, b)
    by_source = defaultdict(list)
    for key, h in original.items():
        by_source[h].append(key)
    variants = {}
    for r in records:
        r.metadata["original_inventory_matches"] = by_source.get(r.sha256, [])
        # Roboflow versions of an explicitly named source image share provenance.
        filename = Path(r.path).name
        if ".rf." in filename:
            key = (r.dataset_id, Path(r.path).parts[0], filename.split(".rf.")[0])
            if key in variants:
                union(r.sha256, variants[key])
            variants[key] = r.sha256
    for r in records:
        r.group_id = "evidence-" + find(r.sha256)
        r.metadata["grouping_reviewed"] = True
        r.metadata["grouping_basis"] = "file/pixel identity, source variants, existing visual review; no temporal inference"
    group_records(records)
    return {str(p.name): file_hash(p) for p in [*inventories, review]}


def prepare_filtered(config_path, runs_root=None):
    config_path = Path(config_path).resolve()
    cfg = read_config(config_path)
    root = (config_path.parent / cfg["storage_root"]).resolve()
    workers = cfg.get("preparation_workers", 16)
    mapping = read_config(root / "configs/benchmark/mappings.yaml")
    taxonomy = read_config(root / "configs/benchmark/taxonomy.yaml")
    mapping["datasets"]["mpid"]["isolador"] = {"target": "isolador", "context_attributes": {"material": "material"}}
    mapping["datasets"]["ttpla"]["tower_tucohy"] = {"exclude": True, "reason": "Outside the NTV lattice-tower task"}
    with Run(Path(runs_root or root / "runs"), "filtered_prepare", cfg) as run:
        all_records, sources, exclusions = [], {}, []
        for name, folder in FOLDERS.items():
            source_root = (root / cfg["filtered_root"] / folder).resolve()
            if name in ("ptl_ai", "mpid"):
                reader, records = read_filtered_yolo(name, source_root, workers)
            else:
                spec = {"id": name, "images": ["images/*"], "workers": workers,
                        "annotations": ["annotations.json"] if name == "stn_plad" else ["labels/*.json"]}
                reader = (CocoReader if name == "stn_plad" else LabelMeReader)(spec, source_root)
                records = reader.read()
            errors = [asdict(i) for i in reader.issues if i.code not in ("image_without_annotations", "missing_source_split")]
            if errors:
                raise ValueError(f"Source {name}: {errors[:5]}")
            if len(records) != cfg["expected_images"][name]:
                raise ValueError(f"Unexpected image count for {name}")
            records = harmonize(records, taxonomy, mapping)
            for r in records:
                r.annotation_scope = dict.fromkeys(CLASSES[name], "exhaustive")
                r.metadata["coverage_confirmation"] = cfg["coverage_confirmation"]
                if name == "ttpla" and any(a.original_label == "void" for a in r.annotations):
                    r.metadata.update(excluded=True, exclusion_reason="approved_void_image_exclusion")
                    exclusions.append({"image_id": r.image_id, "path": r.path, "reason": "void"})
                    continue
                if name == "ttpla":
                    for a in r.annotations:
                        if a.label != "torre" or a.mapping_status != "mapped":
                            continue
                        points = a.geometry.coordinates
                        box = [max(0., min(p[0] for p in points)), max(0., min(p[1] for p in points)),
                               min(float(r.width), max(p[0] for p in points)), min(float(r.height), max(p[1] for p in points))]
                        a.flags["geometry_conversion"] = {"from": asdict(a.geometry), "rule": "axis_aligned_envelope_clip_to_image"}
                        a.geometry = Geometry("bbox", box)
            sources[name] = {"root": str(source_root), "evidence": reader.evidence,
                             "version": digest({r.path: r.sha256 for r in records} | reader.evidence),
                             "config": {"dataset": name, "filtered_folder": folder}}
            all_records.extend(records)
        if len(exclusions) != 12:
            raise ValueError("Expected exactly 12 TTPLA void exclusions")
        evidence = evidence_groups(all_records, root / "metadata/external_datasets/v1")
        splits = {}
        for name in FOLDERS:
            active = [r for r in all_records if r.dataset_id == name and not r.metadata.get("excluded")]
            splits[name] = stratify_groups(active, CLASSES[name], cfg.get("split_seed", 42))
            splits[name]['distribution'] = {}
            for split in ('train', 'val', 'test'):
                selected = [r for r in active if r.split == split]
                boxes = Counter(a.label for r in selected for a in r.annotations
                                if a.mapping_status == 'mapped' and a.label in CLASSES[name])
                positives = Counter(c for r in selected for c in
                                    {a.label for a in r.annotations if a.mapping_status == 'mapped' and a.label in CLASSES[name]})
                splits[name]['distribution'][split] = {
                    'images': len(selected), 'annotations': dict(boxes), 'positive_images': dict(positives),
                    'negative_images': sum(not any(a.label in CLASSES[name] and a.mapping_status == 'mapped'
                                                  for a in r.annotations) for r in selected),
                    'materials': dict(Counter(r.metadata['material'] for r in selected if r.metadata.get('material'))),
                    'members': [{'image_id': r.image_id, 'path': r.path, 'sha256': r.sha256, 'group_id': r.group_id}
                                for r in sorted(selected, key=lambda r: r.image_id)]}
        ntv_spec = read_config(root / "configs/benchmark/ntv-ai.yaml")
        ntv_spec["root"] = str((root / cfg["ntv_root"]).resolve())
        ntv_spec["workers"] = workers
        reader = YoloReader(ntv_spec, Path(ntv_spec["root"]))
        ntv = reader.read()
        identity = {r.path: r.sha256 for r in ntv} | reader.evidence
        if reader.issues or digest(identity) != cfg["ntv_input_digest"]:
            raise ValueError("Original NTV lot identity or labels differ from the approved pilot")
        ntv = temporal_pilot(harmonize(ntv, taxonomy, mapping))
        if sum(r.split == 'test' for r in ntv) != 8 or sum(len(r.annotations) for r in ntv if r.split == 'test') != 138:
            raise ValueError('NTV test must retain exactly eight images and 138 annotations')
        for r in ntv:
            r.annotation_scope = dict.fromkeys(CLASSES["ntv_ai"], "exhaustive")
        sources["ntv_ai"] = {"root": ntv_spec["root"], "evidence": reader.evidence,
                             "version": digest(identity), "config": ntv_spec}
        all_records.extend(ntv)
        # Cross-source exact copies must not become independent evaluation targets.
        for field in ("sha256", "pixel_sha256"):
            identities = defaultdict(set)
            for r in all_records:
                if not r.metadata.get("excluded"):
                    identities[getattr(r, field)].add(r.dataset_id)
            if any(len(ds) > 1 for key, ds in identities.items() if key):
                raise ValueError("Exact overlap between sources requires explicit shared grouping")
        catalog = {"purpose": "exploratory", "datasets": {}, "preparation_run": run.id, "storage_root": str(root)}
        lineage = {"operation": "filtered_prepare", "mapping_sha256": digest(mapping),
                   "taxonomy_sha256": digest(taxonomy), "historical_evidence": evidence,
                   "coverage_confirmation": cfg["coverage_confirmation"], "split_policy": "30:8:8; no external temporal embargo"}
        output = root / "data/prepared" / run.id
        for name in CLASSES:
            rows = [r for r in all_records if r.dataset_id == name]
            active = [r for r in rows if not r.metadata.get("excluded")]
            problems = eligibility(active, CLASSES[name])
            if problems:
                atomic_json(run.path / f"{name}-issues.json", problems)
                raise ValueError(f"{name} has eligibility failures: {problems[:3]}")
            catalog["datasets"][name] = {}
            for protocol in ("full640", "tiles1280"):
                transform = {"enabled": protocol == "tiles1280", "size": 1280, "overlap": .25, "border_policy": "clip"}
                snapshot_sources = {name: deepcopy(sources[name])}
                stored = deepcopy(rows)
                samples = active
                folder = output / name / protocol
                if transform["enabled"]:
                    samples = materialize_tiles(active, snapshot_sources, CLASSES[name], transform,
                                                output / name / "tiles", workers=workers)
                    stored.extend(samples)
                view = {"id": f"{name}-filtered-{protocol}-v1", "classes": CLASSES[name],
                        "image_ids": [r.image_id for r in samples], "evaluation_image_ids": [r.image_id for r in active],
                        "eligible": True, "purpose": "exploratory", "splits": {"mode": "explicit"}, "tiling": transform,
                        "independence": "within_capture_exploratory" if name == "ntv_ai" else "reviewed_filtered_subset"}
                manifest = write_snapshot(folder, stored, snapshot_sources, lineage, [], view=view)
                atomic_json(folder / "splits.json", {s: [r.image_id for r in active if r.split == s] for s in ("train", "val", "test")})
                atomic_json(folder / "dataset.json", descriptor(manifest))
                if transform["enabled"]:
                    atomic_json(folder / "tile-impact.json", tile_impact(active, samples, CLASSES[name]))
                entry = {**descriptor(manifest), "eligible": True, "blockers": [], "counts": dict(Counter(r.split for r in active))}
                catalog["datasets"][name][protocol] = entry
        atomic_json(run.path / "splits-report.json", splits)
        atomic_json(run.path / "exclusions.json", exclusions)
        atomic_json(run.path / "resolved-mapping.json", mapping)
        atomic_json(run.path / "datasets.json", catalog)
        return {"run_id": run.id, "catalog": str(run.path / "datasets.json"), "splits": splits}
