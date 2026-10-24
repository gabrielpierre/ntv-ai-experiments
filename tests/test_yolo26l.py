import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import image, save_yaml
from ntv_ai.cli import execute, parser
from ntv_ai.data.adapters import YoloReader
from ntv_ai.domain.records import PreparedDataset
from ntv_ai.models.backends import yolo26l as backend
from ntv_ai.pipelines.yolo26l import run_model
from ntv_ai.storage import file_hash, read_snapshot, write_snapshot


@pytest.mark.parametrize("layout", ["train/images", "images/train"])
def test_yolo_layouts_and_declared_splits(tmp_path, layout):
    image(tmp_path / layout / "a.png")
    labels = tmp_path / layout.replace("images", "labels") / "a.txt"
    labels.parent.mkdir(parents=True)
    labels.write_text("0 0.5 0.5 0.5 0.5\n")
    save_yaml(tmp_path / "data.yaml", {"names": ["isolador"], "val": layout})
    reader = YoloReader({"id": "x", "images": ["**/*.png"],
                         "subsets": [{"name": "x", "yaml": "data.yaml"}]}, tmp_path)
    record = reader.read()[0]
    assert not reader.issues
    assert record.original_splits == [{"split": "val"}]
    assert record.annotations[0].geometry.coordinates == [10, 7.5, 30, 22.5]
    assert record.annotations[0].source["path"] == str(labels.relative_to(tmp_path))


def test_yolo_path_and_split_lists(tmp_path):
    image(tmp_path / "dataset/images/sub/a.png")
    labels = tmp_path / "dataset/labels/sub/a.txt"
    labels.parent.mkdir(parents=True)
    labels.write_text("")
    (tmp_path / "dataset/train.txt").write_text("./images/sub/a.png\n")
    save_yaml(tmp_path / "source.yaml", {"path": "dataset", "train": ["train.txt"], "names": {0: "x"}})
    reader = YoloReader({"id": "x", "images": ["**/*.png"],
                         "subsets": [{"name": "x", "yaml": "source.yaml"}]}, tmp_path)
    r = reader.read()[0]
    assert r.original_splits == [{"split": "train"}]
    assert "dataset/train.txt" in reader.evidence
    assert {i.code for i in reader.issues} == {"image_without_annotations"}


def test_declared_split_does_not_include_unlisted_images(tmp_path):
    image(tmp_path / "images/train/a.png")
    image(tmp_path / "images/train/b.png", "red")
    (tmp_path / "labels/train").mkdir(parents=True)
    for stem in ("a", "b"):
        (tmp_path / f"labels/train/{stem}.txt").write_text("0 .5 .5 .5 .5")
    (tmp_path / "train.txt").write_text("./images/train/a.png\n./images/train/missing.png\n")
    save_yaml(tmp_path / "data.yaml", {"train": "train.txt", "names": ["x"]})
    reader = YoloReader({"id": "x", "images": ["**/*.png"],
                         "subsets": [{"name": "x", "yaml": "data.yaml"}]}, tmp_path)
    rows = reader.read()
    assert next(r for r in rows if r.path.endswith("b.png")).original_splits == []
    assert {i.code for i in reader.issues} == {"missing_source_split", "image_reference_unresolved"}


def test_mpid_legacy_relative_split(tmp_path):
    image(tmp_path / "train/images/a.png")
    (tmp_path / "train/labels").mkdir()
    (tmp_path / "train/labels/a.txt").write_text("0 .5 .5 .1 .1")
    save_yaml(tmp_path / "data.yaml", {"train": "../train/images", "names": ["1"]})
    reader = YoloReader({"id": "mpid", "images": ["**/*.png"],
                         "subsets": [{"name": "glass", "yaml": "data.yaml", "material": "vidro"}]}, tmp_path)
    r = reader.read()[0]
    assert not reader.issues
    assert r.metadata["material"] == "vidro"
    assert r.original_splits == [{"split": "train"}]


@pytest.fixture
def prepared(project):
    result = execute(parser().parse_args(["prepare", "--config", str(project["prepare"])]))
    path = Path(result["manifest"])
    manifest, rows = read_snapshot(path)
    dataset = PreparedDataset(backend.artifact(path), manifest["view"]["id"], tuple(manifest["view"]["classes"]))
    return project, dataset, manifest, rows


def test_rejects_ineligible_and_incompatible_classes(project, prepared):
    _, dataset, _, _ = prepared
    with pytest.raises(ValueError, match="Classes"):
        backend.prepared_dataset(replace(dataset, classes=("torre",)))
    raw = PreparedDataset(backend.artifact(project["manifest"]), "bbox", ("isolador",))
    with pytest.raises(ValueError, match="inelegível"):
        backend.YOLO26LBackend().train(raw, {}, project["root"] / "output")


def test_training_checks_integrity_and_does_not_repair(prepared):
    project, dataset, _, _ = prepared
    path = project["raw"] / "train/1.png"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Integridade"):
        backend.prepared_dataset(dataset)
    assert path.read_bytes() == b"changed"


def test_framework_cannot_repair_original_jpeg(tmp_path):
    from ntv_ai.domain.records import ImageRecord
    path = image(tmp_path / "a.jpg")
    path.write_bytes(path.read_bytes()[:-2])
    before = file_hash(path)
    with pytest.raises(ValueError, match="reparação automática bloqueada"):
        backend.verify_framework_images([ImageRecord("a", "fixture", "a.jpg")],
                                        {"fixture": {"root": str(tmp_path)}})
    assert file_hash(path) == before


def test_training_requires_validation(prepared):
    project, _, manifest, records = prepared
    rows = [r for r in records if r.split == "train"]
    view = {**manifest["view"], "image_ids": [r.image_id for r in rows]}
    path = write_snapshot(project["root"] / "train-only", rows, manifest["sources"], {}, [], view=view)
    dataset = PreparedDataset(backend.artifact(path), view["id"], tuple(view["classes"]))
    with pytest.raises(ValueError, match="validação"):
        backend.YOLO26LBackend().train(dataset, {}, project["root"] / "out")


class Values:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def tolist(self): return self.values


class Model:
    names = {0: "isolador"}
    predictor = SimpleNamespace(device=SimpleNamespace(type="cuda"), args=SimpleNamespace(nms=False))
    def predict(self, image, **kwargs):
        assert kwargs["nms"] is False
        return [SimpleNamespace(orig_shape=(image.height, image.width), boxes=SimpleNamespace(
            xyxy=Values([]), conf=Values([]), cls=Values([])))]


@contextmanager
def no_cuda(output, device):
    yield {}


def prediction_config(project, manifest):
    return {"classes": ["isolador"], "sources": manifest["sources"],
            "output": str(project["root"] / "predict"),
            "inference": {"nms": False, "device": 0, "conf": .001, "max_det": 100}}


def test_empty_predictions_and_input_preservation(prepared, monkeypatch):
    project, _, manifest, records = prepared
    before = {p: file_hash(p) for p in project["raw"].rglob("*") if p.is_file()}
    checkpoint = project["root"] / "weights/selected.pt"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(backend, "cuda_runtime", no_cuda)
    monkeypatch.setattr(backend, "load_model", lambda p: Model())
    rows = backend.YOLO26LBackend().predict(backend.artifact(checkpoint), records, prediction_config(project, manifest))
    assert {r.image_id for r in rows} == {r.image_id for r in records}
    assert all(r.detections == [] and r.status == "processed" for r in rows)
    assert before == {p: file_hash(p) for p in before}


def test_prediction_rejects_checkpoint_classes(prepared, monkeypatch):
    project, _, manifest, records = prepared
    checkpoint = project["root"] / "weights/wrong.pt"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(backend, "cuda_runtime", no_cuda)
    monkeypatch.setattr(backend, "load_model", lambda p: SimpleNamespace(names={0: "person"}))
    with pytest.raises(ValueError, match="Classes do checkpoint"):
        backend.YOLO26LBackend().predict(backend.artifact(checkpoint), records, prediction_config(project, manifest))


def test_tiled_predictions_are_reprojected_and_consolidated(prepared, monkeypatch):
    project, _, manifest, records = prepared
    class BoxesModel(Model):
        def predict(self, image, **kwargs):
            return [SimpleNamespace(orig_shape=(image.height, image.width), boxes=SimpleNamespace(
                xyxy=Values([[1, 2, 10, 20], [1, 2, 10, 20]]), conf=Values([.9, .8]), cls=Values([0, 0])))]
    checkpoint = project["root"] / "weights/selected.pt"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(backend, "cuda_runtime", no_cuda)
    monkeypatch.setattr(backend, "load_model", lambda p: BoxesModel())
    record = records[0]
    record.metadata["transform"] = {"parent_image_id": "parent", "window": [10, 20, 50, 50],
                                    "scale": [1, 1], "padding": [0, 0]}
    result = backend.YOLO26LBackend().predict(backend.artifact(checkpoint), [record], prediction_config(project, manifest))
    assert len(result) == 1 and result[0].image_id == "parent"
    assert len(result[0].detections) == 1
    assert result[0].detections[0].bbox_xyxy == [11, 22, 20, 40]


@pytest.mark.parametrize("boxes", [([-1, 0, 10, 20], .8, 0), ([1, 1, 10, 10], .8, 2),
                                 ([1, 1, 10, 10], float("nan"), 0)])
def test_predictions_do_not_silently_correct_invalid_outputs(prepared, boxes):
    coords, conf, cls = boxes
    _, _, _, records = prepared
    output = SimpleNamespace(xyxy=Values([coords]), conf=Values([conf]), cls=Values([cls]))
    with pytest.raises(ValueError):
        backend.normalize_boxes(output, ["isolador"], records[0])


def test_failure_record_and_no_overwrite(prepared, monkeypatch):
    project, dataset, _, _ = prepared
    checkpoint = project["root"] / "weights/initial.pt"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"fixture")
    config = save_yaml(project["root"] / "model.yaml", {
        "manifest": dataset.manifest.path, "checkpoint": str(checkpoint), "runs": "model-runs",
        "training": {"device": 0, "batch": 2}})
    def failure(*args): raise RuntimeError("CUDA indisponível")
    monkeypatch.setattr(backend, "cuda_runtime", failure)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="CUDA"):
            run_model("train", config)
    statuses = [json.loads(p.read_text()) for p in (project["root"] / "model-runs").glob("*/status.json")]
    assert len(statuses) == 2 and all(s["state"] == "failed" for s in statuses)
    assert len({s["run_id"] for s in statuses}) == 2
    for export in (project["root"] / "model-runs").glob("*/backend/export/yolo/images/train/*"):
        assert export.is_symlink()
    assert not list(project["raw"].rglob("*.cache"))


def test_import_does_not_load_frameworks_or_write_files(tmp_path):
    source = "import sys; import ntv_ai.models.backends.yolo26l; import ntv_ai.pipelines.yolo26l; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
    subprocess.run([sys.executable, "-c", source], cwd=tmp_path, check=True)
    assert not list(tmp_path.iterdir())
