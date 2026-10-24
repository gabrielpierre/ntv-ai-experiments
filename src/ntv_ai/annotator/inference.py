"""Resumable proposals in a new directory; never writes to human annotation sources."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

from ntv_ai.annotator.data import CLASSES, protected_hashes, read_reviews
from ntv_ai.experiments.config import read_config
from ntv_ai.storage import atomic_json, digest, file_hash, inside, read_snapshot, safe_output


def current_protection(cfg):
    protected = set()
    for source in cfg["review_sources"]:
        path = Path(source["coco"])
        coco = json.loads(path.read_text(encoding="utf-8"))
        protected |= protected_hashes(coco, read_reviews(source["database"], file_hash(path)))
    _, records = read_snapshot(Path(cfg["human_manifest"]))
    protected |= {r.sha256 for r in records if not r.metadata.get("transform")}
    return protected


def initialize(cfg):
    selection = json.loads(Path(cfg["selection"]).read_text(encoding="utf-8"))
    if selection.get("selection_sha256") != digest({k: v for k, v in selection.items() if k != "selection_sha256"}):
        raise ValueError("Selection report changed; generate a new version")
    if set(selection["routes"]) != set(CLASSES) or any(v is None for v in selection["routes"].values()):
        raise ValueError("All seven classes require a supported operating point")
    for label, override in cfg.get("class_overrides", {}).items():
        if label not in CLASSES or set(override) - {"candidate", "threshold"}:
            raise ValueError("Invalid class override")
        selection["routes"][label].update(override)
        route = selection["routes"][label]
        if route["candidate"] not in selection["candidates"] or not math.isfinite(route["threshold"]) or not 0.001 <= route["threshold"] <= 1:
            raise ValueError("Invalid candidate/threshold override")
    if not 0 < cfg.get("nms_iou", 0.5) <= 1:
        raise ValueError("Invalid NMS IoU")
    root = Path(cfg["image_root"]).resolve()
    out = Path(cfg["output"]).resolve()
    protected_paths = [root, Path(cfg["human_manifest"]).parent, Path(cfg["selection"]).parent]
    for source in cfg["review_sources"]:
        protected_paths += [Path(source["coco"]).parent, Path(source["database"]).parent]
    safe_output(out, protected_paths)
    identity = {"configuration": cfg, "selection_sha256": selection["selection_sha256"],
                "inventory_sha256": file_hash(Path(cfg["inventory"])),
                "human_manifest_sha256": file_hash(Path(cfg["human_manifest"]))}
    identity["version"] = digest(identity)
    if (out / "identity.json").exists():
        if json.loads((out / "identity.json").read_text()) != identity:
            raise ValueError("Incompatible resume; choose a new output directory")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Refusing to adopt a nonempty output directory")
        atomic_json(out / "identity.json", identity)
    inventory = json.loads(Path(cfg["inventory"]).read_text(encoding="utf-8"))
    excluded = current_protection(cfg)
    frozen_path = out / "inventory.json"
    if frozen_path.exists():
        images = json.loads(frozen_path.read_text())
    else:
        images = [{"id": i+1, "sha256": row["source_sha256"], "file_name": row["copy_relative_path"],
                   "capture": row["capture"], "width": row["width"], "height": row["height"],
                   "datetime": row.get("datetime")}
                  for i, row in enumerate(sorted((r for r in inventory if r.get("kind") == "image"),
                                                  key=lambda r: (r["capture"], r["copy_relative_path"])))
                  if row["source_sha256"] not in excluded]
        atomic_json(frozen_path, images)
        atomic_json(out / "protected-at-start.json", sorted(excluded))
    for name in {v["candidate"] for v in selection["routes"].values()}:
        candidate = selection["candidates"][name]
        if file_hash(Path(candidate["checkpoint"]["path"])) != candidate["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint hash changed")
    return out, images, selection, identity


def merge_image(image, selection, raw, nms_iou, identity):
    from ntv_ai.domain.records import Detection
    from ntv_ai.transforms.geometry import nms
    result = []
    for label, route in selection["routes"].items():
        candidate = selection["candidates"][route["candidate"]]
        detections = [Detection(**d) for d in raw[route["candidate"]]
                      if d["class_id"] == label and d["score"] >= route["threshold"]]
        for d in nms(detections, nms_iou):
            result.append({**asdict(d), "provenance": {"candidate": route["candidate"],
                "model": candidate["config"]["model"]["backend"], "checkpoint": candidate["checkpoint"],
                "configuration_sha256": digest(candidate["config"]), "threshold": route["threshold"],
                "confidence": d.score, "capture": image["capture"], "pipeline_version": identity["version"]}})
    return {"image": image, "detections": result, "review_stage": "automatic", "status": "processed"}


def infer(config_path, limit=None):
    cfg = read_config(config_path)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    if limit is None and not cfg.get("mass_inference_authorized", False):
        raise ValueError("Mass inference disabled. Review selection/workload and authorize it explicitly.")
    if limit is not None and limit > cfg.get("pilot_limit", 30) and not cfg.get("mass_inference_authorized", False):
        raise ValueError("Limit exceeds the configured small pilot")
    out, images, selection, identity = initialize(cfg)
    protected = current_protection(cfg)
    todo = [im for im in images if im["sha256"] not in protected and not (out/"completed"/(im["sha256"]+".json")).exists()]
    if limit:
        todo = todo[:limit]
    batch_size = cfg.get("batch_size", 50)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    candidates = sorted({r["candidate"] for r in selection["routes"].values()})
    # A single orchestrator per output, even across SSH sessions.
    import fcntl
    with (out / ".orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for start in range(0, len(todo), batch_size):
            protected = current_protection(cfg)
            batch = [im for im in todo[start:start+batch_size] if im["sha256"] not in protected]
            for im in batch:
                if file_hash(inside(Path(cfg["image_root"]), im["file_name"])) != im["sha256"]:
                    raise ValueError("Image changed before resuming cached proposals")
            for name in candidates:
                candidate_key = digest([name, selection["candidates"][name]])
                raw_dir = out / "raw" / candidate_key
                needed = [im for im in batch if not (raw_dir/(im["sha256"]+".json")).exists()]
                if not needed:
                    continue
                worker_dir = out / "workers" / uuid.uuid4().hex
                request = {"candidate": selection["candidates"][name], "images": needed,
                           "image_root": cfg["image_root"], "output": str(raw_dir),
                           "worker_output": str(worker_dir), "reserve_mib": cfg.get("reserve_mib", 2048)}
                atomic_json(worker_dir / "request.json", request)
                env = dict(os.environ)
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
                python = request["candidate"]["config"]["runtime"]["python"]
                with (worker_dir / "worker.log").open("w") as log:
                    subprocess.run([python, "-m", "ntv_ai.annotator.inference", str(worker_dir/"request.json")],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            protected = current_protection(cfg)
            for im in batch:
                if im["sha256"] in protected:
                    continue
                raw = {name: json.loads((out/"raw"/digest([name, selection["candidates"][name]])/(im["sha256"]+".json")).read_text())["detections"]
                       for name in candidates}
                destination = out / "completed" / (im["sha256"] + ".json")
                if not destination.exists():
                    atomic_json(destination, merge_image(im, selection, raw, cfg.get("nms_iou", 0.5), identity))
    return status(config_path)


def status(config_path):
    cfg = read_config(config_path); out = Path(cfg["output"])
    images = json.loads((out / "inventory.json").read_text())
    protected = current_protection(cfg)
    done = {p.stem for p in (out/"completed").glob("*.json")}
    eligible = {im["sha256"] for im in images} - protected
    return {"output": str(out), "eligible": len(eligible), "completed": len(eligible & done),
            "remaining": len(eligible-done), "newly_protected": len({im["sha256"] for im in images} & protected)}


def export(config_path):
    cfg = read_config(config_path)
    out, images, selection, identity = initialize(cfg)
    protected = current_protection(cfg)
    export_dir = out / "exports" / uuid.uuid4().hex
    accepted, annotations, provenance = [], [], []
    for im in images:
        source = out / "completed" / (im["sha256"] + ".json")
        if im["sha256"] in protected or not source.exists():
            continue
        if file_hash(inside(Path(cfg["image_root"]), im["file_name"])) != im["sha256"]:
            raise ValueError("Image changed before export")
        row = json.loads(source.read_text())
        if row["image"] != im or row["status"] != "processed":
            raise ValueError("Invalid completion journal")
        accepted.append({**im, "review_status": "pending_human_review", "review_stage": "automatic"})
        lines = []
        for d in row["detections"]:
            x1, y1, x2, y2 = d["bbox_xyxy"]; w, h = x2-x1, y2-y1
            category = CLASSES.index(d["class_id"])+1
            annotation = {"id": len(annotations)+1, "image_id": im["id"], "category_id": category,
                          "bbox": [x1, y1, w, h], "area": w*h, "iscrowd": 0, "segmentation": [],
                          "score": d["score"], "attributes": {"review_stage": "automatic", **d["provenance"]}}
            annotations.append(annotation)
            provenance.append({"image_sha256": im["sha256"], "annotation_id": annotation["id"], **d["provenance"]})
            lines.append(f"{category-1} {(x1+w/2)/im['width']:.10f} {(y1+h/2)/im['height']:.10f} {w/im['width']:.10f} {h/im['height']:.10f}")
        path = export_dir / "yolo/labels" / Path(im["file_name"]).with_suffix(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    document = {"info": {"purpose": "automatic_proposals_not_ground_truth", "pipeline_version": identity["version"],
                          "selection_sha256": selection["selection_sha256"]},
                "categories": [{"id": i+1, "name": c} for i, c in enumerate(CLASSES)],
                "images": accepted, "annotations": annotations}
    # Re-check before publication. A race can only cancel this separate export, never overwrite a human file.
    if {im["sha256"] for im in accepted} & current_protection(cfg):
        raise RuntimeError("Human changes during export; discard unpublished export and run export again")
    atomic_json(export_dir / "provenance.json", provenance)
    cvat = deepcopy(document)
    for ann in cvat["annotations"]:
        ann["attributes"] = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in ann["attributes"].items()}
    atomic_json(export_dir / "annotations/instances_default.json", cvat)
    atomic_json(export_dir / "preanotacoes.coco.json", document)
    (export_dir / "yolo").mkdir(exist_ok=True)
    (export_dir / "yolo/classes.txt").write_text("\n".join(CLASSES)+"\n", encoding="utf-8")
    atomic_json(export_dir / "export-complete.json", {"images": len(accepted), "boxes": len(annotations), "source": identity})
    return {"export": str(export_dir), "images": len(accepted), "boxes": len(annotations)}


def worker(request_path):
    from PIL import Image
    from ntv_ai.models.registry import backend
    from ntv_ai.models.runtime import gpu_slot, runtime, verify_implementation
    from ntv_ai.models.inference import predict_image
    request = json.loads(Path(request_path).read_text())
    candidate = request["candidate"]
    config = deepcopy(candidate["config"])
    config["classes"] = CLASSES
    output = Path(request["output"])
    checkpoint = candidate["checkpoint"]
    if file_hash(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("Checkpoint changed")
    verify_implementation(config)
    with gpu_slot(config["runtime"]["device"]), runtime(config, Path(request["worker_output"])):
        import torch
        free, _ = torch.cuda.mem_get_info()
        if free < request["reserve_mib"] * 1024**2:
            raise RuntimeError("Insufficient free GPU memory; existing services were left running")
        predictor = backend(config["model"]["backend"]).load_predictor(Path(checkpoint["path"]), config)
        manifest, _ = read_snapshot(Path(config["dataset"]["manifest"]["path"]))
        tiled = manifest["view"]["tiling"]
        tiling = {**tiled, "tiling": tiled.get("enabled", False)}
        for im in request["images"]:
            destination = output / (im["sha256"] + ".json")
            if destination.exists():
                continue
            path = inside(Path(request["image_root"]), im["file_name"])
            if file_hash(path) != im["sha256"]:
                raise ValueError("Source image changed")
            with Image.open(path) as image:
                if image.size != (im["width"], im["height"]):
                    raise ValueError("Source dimensions changed")
                boxes, timings = predict_image(predictor, image.convert("RGB"), tiling, config["protocol"]["tile_iou"])
            atomic_json(destination, {"image_sha256": im["sha256"], "detections": [asdict(d) for d in boxes],
                                      "timings": timings, "checkpoint": checkpoint})


if __name__ == "__main__":
    worker(sys.argv[1])
