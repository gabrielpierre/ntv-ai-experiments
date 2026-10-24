from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from PIL import Image

from ntv_ai.annotator import inference
from ntv_ai.annotator.data import CLASSES
from ntv_ai.domain.records import ImageRecord
from ntv_ai.storage import atomic_json, digest, file_hash, write_snapshot


@pytest.fixture
def setup(tmp_path):
    images = tmp_path/"images"; images.mkdir()
    inventory = []
    for i in range(3):
        path = images/f"{i}.jpg"; Image.new("RGB", (20, 20), (i*70, 30, 40)).save(path)
        inventory.append({"kind": "image", "capture": "001", "source_sha256": file_hash(path),
                          "copy_relative_path": path.name, "width": 20, "height": 20})
    atomic_json(tmp_path/"inventory.json", inventory)
    coco = {"images": [{"id": i, "sha256": r["source_sha256"]} for i, r in enumerate(inventory) if i], "annotations": []}
    coco_path = tmp_path/"reviews/source.json"; atomic_json(coco_path, coco)
    db_path = tmp_path/"reviews/state.sqlite3"
    db = sqlite3.connect(db_path)
    db.executescript("CREATE TABLE metadata(key TEXT,value TEXT); CREATE TABLE reviews(image_id INTEGER,revision INTEGER,state TEXT); CREATE TABLE events(id INTEGER);")
    db.execute("INSERT INTO metadata VALUES ('source_sha256',?)", (file_hash(coco_path),))
    for i in (1, 2):
        db.execute("INSERT INTO reviews VALUES (?,0,?)", (i, json.dumps({"image_id": i, "status": "pending", "boxes": []})))
    db.commit(); db.close()
    r = ImageRecord("human", "human", "0.jpg", 20, 20, inventory[0]["source_sha256"])
    manifest = write_snapshot(tmp_path/"human", [r], {"human": {"root": str(images)}}, {}, [], view={"eligible": True})
    checkpoint = tmp_path/"weights/fake.pt"; checkpoint.parent.mkdir(); checkpoint.write_bytes(b"fixture only")
    selection = {"routes": {c: {"candidate": "fixture", "threshold": .1} for c in CLASSES},
                 "candidates": {"fixture": {"checkpoint": {"path": str(checkpoint), "sha256": file_hash(checkpoint)},
                     "config": {"runtime": {"python": "fixture"}, "model": {"backend": "fixture"}}}}}
    selection["selection_sha256"] = digest(selection)
    atomic_json(tmp_path/"selection/report.json", selection)
    cfg = {"selection": str(tmp_path/"selection/report.json"), "image_root": str(images), "output": str(tmp_path/"output"),
           "inventory": str(tmp_path/"inventory.json"), "human_manifest": str(manifest), "batch_size": 50,
           "review_sources": [{"coco": str(coco_path), "database": str(db_path)}], "mass_inference_authorized": False}
    config = tmp_path/"config.json"; atomic_json(config, cfg)
    return config, cfg, inventory, db_path


def test_mass_disabled_and_input_config_identity(setup):
    path, cfg, _, _ = setup
    with pytest.raises(ValueError, match="Mass inference disabled"):
        inference.infer(path)
    inference.initialize(cfg)
    changed = {**cfg, "batch_size": 25}
    with pytest.raises(ValueError, match="Incompatible resume"):
        inference.initialize(changed)


def test_interrupted_per_image_worker_resumes_and_protects_human_changes(setup, monkeypatch):
    path, cfg, inventory, db_path = setup
    calls = []
    def fake_run(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        calls.append([im["sha256"] for im in request["images"]])
        for i, im in enumerate(request["images"]):
            atomic_json(Path(request["output"])/(im["sha256"]+".json"), {"detections": [
                {"class_id": "torre", "bbox_xyxy": [1, 1, 8, 8], "score": .9},
                {"class_id": "torre", "bbox_xyxy": [1, 1, 8, 8], "score": .8}]})
            if len(calls) == 1:
                raise RuntimeError("simulated power interruption")
    monkeypatch.setattr(inference.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="interruption"):
        inference.infer(path, limit=2)
    result = inference.infer(path, limit=2)
    assert result["completed"] == 2
    assert len(calls[0]) == 2 and len(calls[1]) == 1
    # A human starts editing after prediction: export must now exclude that image.
    db = sqlite3.connect(db_path); db.execute("UPDATE reviews SET revision=1 WHERE image_id=1"); db.commit(); db.close()
    exported = inference.export(path)
    coco = json.loads((Path(exported["export"])/"preanotacoes.coco.json").read_text())
    assert exported["images"] == 1 and exported["boxes"] == 1
    assert coco["images"][0]["sha256"] == inventory[2]["source_sha256"]
    assert coco["annotations"][0]["attributes"]["review_stage"] == "automatic"
    assert (Path(exported["export"])/"annotations/instances_default.json").is_file()


def test_class_threshold_override_is_versioned(setup):
    _, cfg, _, _ = setup
    cfg["class_overrides"] = {"torre": {"threshold": .8}}
    _, _, selection, _ = inference.initialize(cfg)
    assert selection["routes"]["torre"]["threshold"] == .8
