import copy
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from ntv_ai.cli import execute, parser
from ntv_ai.domain.records import PreparedDataset
from ntv_ai.evaluation.bbox import evaluate
from ntv_ai.experiments.recipes import compatibility, resolve_recipe, validate_recipe
from ntv_ai.experiments.selection import CheckpointSelection
from ntv_ai.models.common import artifact, prepared_dataset
from ntv_ai.models.exports import export_dataset
from ntv_ai.storage import load_json, read_snapshot


def recipe():
    return {"schema_version": 1, "purpose": "software_smoke_only", "dataset": {"fixture": "coco8"},
            "model": {"backend": "dfine", "variant": "s", "weights": {"path": "weights", "sha256": "a" * 64}},
            "protocol": {"id": "test", "resolution": 320, "max_detections": 100, "checkpoint_metric": "mAP50_95",
                         "selection_split": "val", "weight_policy": "ema", "confidence": .001, "tile_iou": .5},
            "training": {"epochs": 1, "batch": 1, "workers": 0, "seed": 42, "amp": True, "native": {}},
            "runtime": {"environment": "test", "device": 0, "storage_root": "."}}


def prepared(project):
    result = execute(parser().parse_args(["prepare", "--config", str(project["prepare"])]))
    path = Path(result["manifest"])
    manifest, records = read_snapshot(path)
    data = PreparedDataset(artifact(path), manifest["view"]["id"], tuple(manifest["view"]["classes"]))
    return data, manifest, records


def test_registry_does_not_load_frameworks():
    code = "from ntv_ai.models.registry import BACKENDS; import sys; assert not {'torch', 'ultralytics', 'rfdetr'} & sys.modules.keys(); assert set(BACKENDS) == {'yolo26', 'dfine', 'rfdetr'}"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("section,key,value", [
    ("model", "variant", "x"), ("model", "backend", "unknown"), ("protocol", "resolution", 321),
    ("protocol", "selection_split", "test"), ("protocol", "weight_policy", "best_native"),
    ("protocol", "max_detections", 1), ("training", "batch", -1), ("training", "batch", "auto"),
    ("training", "epochs", True), ("training", "amp", "true"), ("runtime", "device", "cpu"),
])
def test_invalid_recipe_rejected(section, key, value):
    config = recipe()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_recipe(config)


def test_software_fixture_is_not_scientific():
    config = recipe()
    config["purpose"] = "scientific"
    with pytest.raises(ValueError, match="científico"):
        validate_recipe(config)


def test_compatibility_preserves_scientific_differences():
    original = recipe()
    moved = copy.deepcopy(original)
    moved["runtime"]["storage_root"] = "/other/disk"
    moved["model"]["weights"]["path"] = "moved.pt"
    moved["training"]["seed"] = 44
    assert compatibility(original) == compatibility(moved)
    moved['inference_checkpoint'] = {'path':'seed-44/best.pt', 'sha256':'c'*64}
    assert compatibility(original) == compatibility(moved)
    for section, key, value in [("protocol", "resolution", 640), ("training", "batch", 2),
                                ("model", "weights", {"sha256": "b" * 64}),
                                ("dataset", "version", "changed"), ("training", "native", {"lr": .3})]:
        changed = copy.deepcopy(original)
        changed[section][key] = value
        assert compatibility(original) != compatibility(changed)


def test_layers_resolve_and_hash(tmp_path):
    config = recipe()
    (tmp_path / "model.json").write_text(json.dumps(config["model"]))
    config["model"] = "model.json"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(config))
    result = resolve_recipe(path, tmp_path / "disk")
    assert result["model"]["backend"] == "dfine"
    assert result["runtime"]["storage_root"] == str(tmp_path / "disk")
    assert result["provenance"]["layers"]["model"] == artifact(tmp_path / "model.json").sha256


def test_selection_ties_and_resume(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"first")
    selection = CheckpointSelection(tmp_path / "selection")
    selection.consider(0, .5, checkpoint)
    checkpoint.write_bytes(b"tie")
    selection.consider(1, .5, checkpoint)
    assert Path(selection.best["checkpoint"]["path"]).read_bytes() == b"first"
    resumed = CheckpointSelection(tmp_path / "child", tmp_path / "selection")
    resumed.consider(2, .4, checkpoint)
    assert resumed.best["epoch"] == 0
    checkpoint.write_bytes(b"better")
    resumed.consider(3, .6, checkpoint)
    assert resumed.best["epoch"] == 3
    assert Path(selection.best["checkpoint"]["path"]).read_bytes() == b"first"
    with pytest.raises(ValueError):
        resumed.consider(4, float("nan"), checkpoint)
    with pytest.raises(ValueError):
        resumed.consider(3, .9, checkpoint)


def test_portable_sources_and_native_exports(project, tmp_path):
    dataset, manifest, records = prepared(project)
    moved = tmp_path / "new-storage"
    shutil.copytree(project["raw"], moved)
    remapped, selected = prepared_dataset(dataset, {"fixture": str(moved)})
    assert remapped["version"] == manifest["version"]
    assert remapped["sources"]["fixture"]["root"] == str(moved)
    for format in ("coco_zero_based", "roboflow_coco"):
        folder = export_dataset(selected, remapped["sources"], dataset.classes, tmp_path / format, format)
        doc = load_json(folder / ("train.json" if format == "coco_zero_based" else "train/_annotations.coco.json"))
        assert doc["categories"][0]["id"] == 0
        assert doc["annotations"][0]["category_id"] == 0
        assert doc["annotations"][0]["bbox"] == [4, 5, 10, 12]
        path = Path(doc["images"][0]["file_name"])
        if not path.is_absolute():
            path = folder / "train" / path
        assert path.is_file()
        assert load_json(folder / "class_map.json") == [{"canonical": "isolador", "coco_id": 1, "native_id": 0}]


def test_size_metrics_and_class_recall(project, tmp_path):
    pytest.importorskip("pycocotools")
    _, manifest, records = prepared(project)
    record = next(r for r in records if r.split == "val")
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"image_id": record.image_id, "status": "processed", "detections": [
        {"class_id": "isolador", "bbox_xyxy": [4, 5, 14, 17], "score": .99}]}))
    metrics = evaluate(manifest, records, path, {"split": "val"})
    assert metrics["AP_by_size"]["small"] == pytest.approx(1)
    assert metrics["AP_by_size"]["large"] is None
    assert metrics["per_class"]["isolador"]["recall50"] == pytest.approx(1)
    path.write_text(json.dumps({"image_id": record.image_id, "status": "processed", "detections": []}))
    metrics = evaluate(manifest, records, path, {"split": "val"})
    assert metrics["per_class"]["isolador"]["AR50_95"] == 0
    with pytest.raises(ValueError, match="vazio"):
        evaluate(manifest, [r for r in records if r.split != "val"], path, {"split": "val"})
