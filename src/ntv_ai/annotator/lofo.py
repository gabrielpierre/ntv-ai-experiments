"""LOFO plans and guarded preparation; never launches a scientific experiment."""
from copy import deepcopy
import math
import hashlib
import json
from pathlib import Path

from ntv_ai.annotator.data import CLASSES
from ntv_ai.annotator.splits import grouped_split, support
from ntv_ai.experiments.config import read_config
from ntv_ai.storage import atomic_json, digest, read_snapshot, write_snapshot


def plan(config_path):
    cfg = read_config(config_path)
    captures = cfg["captures"]
    if len(captures) != 3 or len(set(captures)) != 3:
        raise ValueError("LOFO requires three distinct captures")
    result = {"purpose": "same_structure_cross_capture_transfer", "executed": False,
              "initialization": "official_pretrained_only", "seeds": [42, 43, 44],
              "models": ["yolo26-s", "dfine-s", "rfdetr-nano"], "protocols": ["full640", "tiles1280"],
              "jobs": 54, "requires": ["validated_annotations", "independent_test_audit", "explicit_training_authorization"],
              "folds": [{"id": f"test-{c[-3:]}", "train_validation_captures": [x for x in captures if x != c],
                         "test_capture": c} for c in reversed(captures)],
              "split": cfg["split"], "audit_fraction": 0.10,
              "absent_class_metric": None, "hyperparameters_frozen_before_any_test": True}
    atomic_json(Path(cfg["output"]) / "lofo-plan.json", result)
    return result


def check_validation(records):
    for r in records:
        evidence = r.metadata.get("review_evidence", {})
        if r.metadata.get("review_stage") != "validated" or evidence.get("review_stage") != "validated":
            raise ValueError(f"Image lacks explicit second validation: {r.image_id}")
        if not evidence.get("reviewed_by") or not evidence.get("validated_by") or evidence["reviewed_by"] == evidence["validated_by"]:
            raise ValueError("Validation requires distinct identified reviewers")
        if not evidence.get("validated_at") or evidence.get("reviewed_resolution") != "original":
            raise ValueError("Missing original-resolution validation evidence")
        boxes = [(a.original.get("id", a.attributes.get("review_box_id")), CLASSES.index(a.label)+1,
                  [float(a.geometry.coordinates[0]), float(a.geometry.coordinates[1]),
                   float(a.geometry.coordinates[2]-a.geometry.coordinates[0]), float(a.geometry.coordinates[3]-a.geometry.coordinates[1])])
                 for a in r.annotations]
        # Prefer original xywh to avoid subtraction roundoff in exported floating coordinates.
        for i, a in enumerate(r.annotations):
            if a.original.get("bbox"):
                x, y, w, h = a.original["bbox"]
                if any(not math.isclose(v, expected, rel_tol=0, abs_tol=1e-7)
                       for v, expected in zip(a.geometry.coordinates, (x, y, x+w, y+h), strict=True)):
                    raise ValueError("Geometry differs from validated human boxes")
                boxes[i] = (boxes[i][0], boxes[i][1], [float(v) for v in a.original["bbox"]])
        if any(b[0] is None for b in boxes):
            raise ValueError("Missing annotation IDs for validation verification")
        fingerprint = hashlib.sha256(json.dumps(sorted(boxes), ensure_ascii=False, allow_nan=False,
                                                separators=(",", ":")).encode()).hexdigest()
        if evidence.get("validated_annotation_sha256") != fingerprint:
            raise ValueError("Annotations no longer match the second validation")


def preflight_guard(config, manifest, manifest_path):
    """Additional gate for LOFO manifests, leaving historical recipes untouched."""
    if manifest.get("lineage", {}).get("operation") != "lofo":
        return
    root = Path(config["runtime"]["storage_root"])
    backend = config["model"]["backend"]
    model_name = {"yolo26": "yolo26-s", "dfine": "dfine-s", "rfdetr": "rfdetr-nano"}.get(backend)
    if not model_name:
        raise ValueError("Unsupported LOFO model")
    official = read_config(root / f"configs/experiments/models/{model_name}.yaml")["weights"]
    from ntv_ai.storage import file_hash
    weights = config["model"]["weights"]
    if weights["sha256"] != official["sha256"] or file_hash(root / weights["path"]) != official["sha256"]:
        raise ValueError("LOFO requires official initialization, never annotator weights")
    audit_path = Path(manifest["lineage"].get("audit_root", Path(manifest_path).parent.parent)) / "audit-completed.json"
    if not audit_path.exists():
        raise ValueError("LOFO blocked: independent test audit has not been recorded")
    audit = json.loads(audit_path.read_text())
    required = json.loads((audit_path.parent / "audit-required.json").read_text())
    if audit.get("dataset_version") != required["dataset_version"] or audit.get("predictions_visible") is not False:
        raise ValueError("Invalid or stale independent audit")
    _, rows = read_snapshot(Path(manifest_path))
    test = {r.image_id: r for r in rows if r.split == "test"}
    checks = {r["image_id"]: r for r in audit.get("images", [])}
    if not set(required["image_ids"]) <= checks.keys():
        raise ValueError("Independent audit sample is incomplete")
    for key in required["image_ids"]:
        item, r = checks[key], test[key]
        first = r.metadata["review_evidence"]["reviewed_by"]
        if not item.get("reviewer_id") or item["reviewer_id"] == first or not item.get("reviewed_at") or item.get("accepted") is not True:
            raise ValueError("Audit must be signed by a person independent of the initial correction")
        if item.get("annotation_sha256") != r.metadata["review_evidence"]["validated_annotation_sha256"]:
            raise ValueError("Audit references different annotations")


def audit_sample(records, seed=43, fraction=0.10):
    """Deterministic sample covering present classes and valid negatives."""
    ordered = sorted(records, key=lambda r: digest([seed, r.sha256]))
    selected = {}
    for label in CLASSES:
        matching = [r for r in ordered if any(a.label == label for a in r.annotations)]
        if matching:
            selected[matching[0].image_id] = matching[0]
    negatives = [r for r in ordered if not r.annotations]
    if negatives:
        selected[negatives[0].image_id] = negatives[0]
    minimum = max(len(selected), math.ceil(fraction * len(records)))
    for r in ordered:
        if len(selected) >= minimum:
            break
        selected[r.image_id] = r
    return sorted(selected)


def prepare(config_path):
    cfg = read_config(config_path)
    manifest, records = read_snapshot(Path(cfg["validated_manifest"]))
    records = [r for r in records if not r.metadata.get("transform")]
    check_validation(records)
    if len(records) != cfg.get("expected_images", 4427):
        raise ValueError("Complete human review required before final LOFO preparation")
    captures = cfg["captures"]
    if {r.metadata["capture_id"] for r in records} != set(captures):
        raise ValueError("Capture inventory differs from frozen LOFO plan")
    output = Path(cfg["output"]) / "prepared"
    if output.exists():
        raise FileExistsError(output)
    folds = []
    for capture in reversed(captures):
        test = [deepcopy(r) for r in records if r.metadata["capture_id"] == capture]
        training = [deepcopy(r) for r in records if r.metadata["capture_id"] != capture]
        for attr in ("sha256", "pixel_sha256"):
            if ({getattr(r, attr) for r in training} - {""}) & ({getattr(r, attr) for r in test} - {""}):
                raise ValueError("Exact duplicate across held-out capture")
        split = grouped_split(training, cfg["capture_anchors"], CLASSES, cfg["split"])
        for r in test:
            r.split = "test"; r.group_id = f"heldout:{capture}"
            r.metadata.update(grouping_reviewed=True, grouping_basis="heldout_capture", excluded=False)
        rows = training + test
        view = {"id": f"lofo-{capture[-3:]}-full640", "purpose": "scientific", "classes": CLASSES,
                "eligible": True, "image_ids": [r.image_id for r in rows if r.split],
                "evaluation_image_ids": [r.image_id for r in rows if r.split], "tiling": {"enabled": False},
                "splits": {"mode": "leave_one_capture_out", "test_capture": capture}}
        folder = output / f"test-{capture[-3:]}"
        path = write_snapshot(folder / "full640", rows, manifest["sources"],
                              {"operation": "lofo", "parent_version": manifest["version"], "test_capture": capture}, [], view=view)
        sample = audit_sample(test)
        atomic_json(folder / "audit-required.json", {"dataset_version": manifest["version"],
            "image_ids": sample, "status": "pending_independent_audit", "predictions_visible": False})
        atomic_json(folder / "partition.json", split)
        if cfg.get("training_template"):
            from ntv_ai.annotator.training import build_views, write_recipes
            template = read_config(Path(cfg["training_template"]))
            template.update(purpose="scientific", seeds=[42, 43, 44], epochs=cfg.get("epochs", 100))
            views = build_views(path, folder / "protocols", template.get("tile_workers", 4))
            write_recipes(template, views, folder)
        folds.append({"test_capture": capture, "manifest": str(path), "support": support(rows, CLASSES),
                      "audit_required": str(folder / "audit-required.json"), "training_enabled": False})
    atomic_json(output / "folds.json", {"folds": folds, "executed": False})
    return {"folds": folds, "executed": False}
