"""Operating-point selection on development validation, without a precision floor."""
import json
from pathlib import Path
from statistics import mean

from ntv_ai.evaluation.bbox import evaluate
from ntv_ai.experiments.config import read_config
from ntv_ai.storage import atomic_json, digest, file_hash, read_snapshot

THRESHOLDS = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def operating_point(records, predictions, label, threshold, match_iou=0.5):
    tp = fp = fn = positive_images = 0
    counts = []
    for r in records:
        truth = [a.geometry.coordinates for a in r.annotations if a.label == label]
        positive_images += bool(truth)
        detected = sorted([d for d in predictions[r.image_id] if d["class_id"] == label and d["score"] >= threshold],
                          key=lambda d: (-d["score"], d["bbox_xyxy"]))
        counts.append(len(detected)); unmatched = set(range(len(truth)))
        for d in detected:
            best = max(unmatched, key=lambda j: (iou(truth[j], d["bbox_xyxy"]), -j), default=None)
            if best is not None and iou(truth[best], d["bbox_xyxy"]) >= match_iou:
                tp += 1; unmatched.remove(best)
            else:
                fp += 1
        fn += len(unmatched)
    support = tp + fn
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/support if support else None
    f2 = 5*tp/(5*tp+4*fn+fp) if support else None
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f2": f2, "support_objects": support, "support_images": positive_images,
            "proposals": sum(counts), "proposals_per_image": mean(counts) if counts else 0,
            "false_positives_per_image": fp/len(records) if records else 0,
            "maximum_proposals_per_image": max(counts, default=0)}


def choose(rows):
    usable = [r for r in rows if r["f2"] is not None]
    return min(usable, key=lambda r: (-r["f2"], -r["recall"], r["fp"], r["candidate"], r["threshold"])) if usable else None


def select(config_path):
    cfg = read_config(config_path)
    if not cfg.get("candidates"):
        raise ValueError("Register completed development runs before selecting an annotator")
    manifest, all_records = read_snapshot(Path(cfg["manifest"]))
    records = [r for r in all_records if r.split == "val" and not r.metadata.get("transform")]
    classes = manifest["view"]["classes"]
    if not records:
        raise ValueError("Empty development validation")
    truth_identity = digest(sorted((r.sha256, [(a.label, a.geometry.coordinates) for a in r.annotations]) for r in records))
    alternatives = {c: [] for c in classes}
    candidates = {}
    for item in cfg["candidates"]:
        run = Path(item["run"])
        run_config = json.loads((run / "config.json").read_text())
        result = json.loads((run / "result.json").read_text())
        if run_config["purpose"] == "software_smoke_only":
            raise ValueError("Software smoke cannot supply an annotator")
        _, run_records = read_snapshot(Path(run_config["dataset"]["manifest"]["path"]))
        run_val = [r for r in run_records if r.split == "val" and not r.metadata.get("transform")]
        identity = digest(sorted((r.sha256, [(a.label, a.geometry.coordinates) for a in r.annotations]) for r in run_val))
        if identity != truth_identity or {r.image_id for r in run_val} != {r.image_id for r in records}:
            raise ValueError("Candidates must share the exact frozen validation annotations")
        best = result["best"]
        checkpoint = best["checkpoint"]
        if file_hash(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("Selected checkpoint changed")
        prediction_path = run / f"val-{best['epoch']:04d}.jsonl"
        metrics = evaluate(manifest, all_records, prediction_path, {"split": "val", "max_detections": 100})
        rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        predictions = {r["image_id"]: r["detections"] for r in rows}
        name = item["id"]
        if name in candidates:
            raise ValueError("Duplicate candidate ID")
        candidates[name] = {"run": str(run), "config": run_config, "checkpoint": checkpoint,
                            "epoch": best["epoch"], "metrics": metrics,
                            "prediction_sha256": file_hash(prediction_path)}
        for label in classes:
            for threshold in cfg.get("thresholds", THRESHOLDS):
                if not 0.01 <= threshold <= 0.90:
                    raise ValueError("Threshold outside configured development range")
                row = operating_point(records, predictions, label, threshold)
                alternatives[label].append({**row, "candidate": name,
                    "per_capture": {capture: operating_point([r for r in records if r.metadata["capture_id"] == capture],
                                     predictions, label, threshold)
                                    for capture in sorted({r.metadata["capture_id"] for r in records})}})
    routes = {c: choose(rows) for c, rows in alternatives.items()}
    report = {"purpose": "annotator_development_not_independent_test", "manifest": str(cfg["manifest"]),
              "dataset_version": manifest["version"], "validation_identity": truth_identity,
              "matching_iou": 0.5, "beta": 2, "precision_floor": None,
              "candidates": candidates, "routes": routes, "alternatives": alternatives,
              "mass_inference_authorized": False,
              "workload_note": "Counts refer to reviewed validation only; not a guarantee for the full collection."}
    report["selection_sha256"] = digest(report)
    atomic_json(Path(cfg["output"]), report)
    return {"report": cfg["output"], "routes": routes, "mass_inference_executed": False}
