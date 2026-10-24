import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from ntv_ai.domain.records import PredictionRecord
from ntv_ai.experiments import campaigns
from ntv_ai.experiments.recipes import validate_recipe
from ntv_ai.models.common import artifact
from ntv_ai.models.inference import detections
from ntv_ai.pipelines import experiments
from ntv_ai.pipelines.experiment_worker import merged_predictions
from ntv_ai.storage import atomic_json, load_json
from test_experiment_core import prepared, recipe


def local_recipe(project, tmp_path):
    dataset, manifest, records = prepared(project)
    config = recipe()
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"unit-test-only")
    config["model"]["weights"] = asdict(artifact(weight))
    config["runtime"].update(storage_root=str(tmp_path), python=sys.executable)
    config["dataset"] = {"manifest": asdict(dataset.manifest), "version": manifest["version"],
                         "view_id": dataset.view_id, "classes": list(dataset.classes), "source_roots": {}}
    return config


def test_preflight_rejects_ineligible_data_before_worker(project, tmp_path, monkeypatch):
    config = local_recipe(project, tmp_path)
    config["dataset"]["manifest"] = asdict(artifact(project["manifest"]))
    from ntv_ai.storage import read_snapshot
    config["dataset"]["version"] = read_snapshot(project["manifest"])[0]["version"]
    calls = []
    monkeypatch.setattr(experiments.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="inelegível"):
        experiments.launch("train", config)
    assert calls == []


def test_worker_oom_is_recorded_without_changing_recipe(project, tmp_path, monkeypatch):
    config = local_recipe(project, tmp_path)
    original = copy.deepcopy(config)
    real_run = experiments.subprocess.run
    calls = []
    def worker(command, **kwargs):
        if "ntv_ai.pipelines.experiment_worker" not in command:
            return real_run(command, **kwargs)
        request = load_json(Path(command[-1]))
        calls.append(request)
        atomic_json(Path(request["output"]) / "failure.json", {"message": "CUDA out of memory", "oom": True,
                                                               "type": "OutOfMemoryError", "recipe_changed": False})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(experiments.subprocess, "run", worker)
    with pytest.raises(RuntimeError, match="out of memory"):
        experiments.launch("train", config, runs_root=tmp_path / "experiments")
    assert len(calls) == 1
    assert config == original
    folder = Path(calls[0]["output"])
    assert load_json(folder / "status.json")["state"] == "failed"
    assert load_json(folder / "status.json")["failure"]["oom"]
    saved = load_json(folder / "config.json")
    assert saved["training"]["batch"] == 1 and saved["protocol"]["resolution"] == 320


def test_native_controls_cannot_override_protocol():
    for backend, variant, bad_option in [("yolo26", "s", "batch"), ("rfdetr", "nano", "run_test"), ("dfine", "s", "epochs")]:
        config = recipe()
        config["model"].update(backend=backend, variant=variant)
        config["training"]["native"][bad_option] = True
        with pytest.raises(ValueError, match="nativas"):
            validate_recipe(config)


def test_class_and_boundary_contract():
    result = detections([[-1, 0, 11, 20], [12, 2, 15, 4]], [.9, .8], [1, 0], ["a", "b"], (10, 20), .001, 100, clip=True)
    assert len(result) == 1 and result[0].class_id == "b" and result[0].bbox_xyxy == [0, 0, 10, 20]
    for labels, scores in [([2], [.5]), ([.5], [.5]), ([0], [float("nan")])]:
        with pytest.raises(ValueError):
            detections([[0, 0, 2, 2]], scores, labels, ["a", "b"], (10, 10), 0, 100, clip=True)
    assert detections([], [], [], ["a"], (10, 10), 0, 100) == []


def test_resume_links_run_and_requires_same_recipe(project, tmp_path):
    config = experiments.preflight(local_recipe(project, tmp_path))
    parent = tmp_path / "parent"
    parent.mkdir()
    checkpoint = parent / "resume.ckpt"
    checkpoint.write_bytes(b"full-native-state")
    atomic_json(parent / "config.json", config)
    atomic_json(parent / "status.json", {"run_id": "parent", "state": "completed"})
    atomic_json(parent / "result.json", {"resume": asdict(artifact(checkpoint)), "last_epoch": 0})
    extended = copy.deepcopy(config)
    extended["training"]["epochs"] = 2
    resumed = experiments.resume_configuration(extended, parent)
    assert resumed["resume"]["parent_run"] == "parent"
    assert load_json(parent / "config.json")["training"]["epochs"] == 1
    for field, value in [("batch", 2), ("seed", 43)]:
        changed = copy.deepcopy(extended)
        changed["training"][field] = value
        with pytest.raises(ValueError, match="incompatível"):
            experiments.resume_configuration(changed, parent)
    checkpoint.write_bytes(b"modified")
    assert experiments.resume_configuration(extended, parent)["resume"]["parent_run"] == "parent"
    checkpoint.unlink()
    with pytest.raises(ValueError, match="Arquivo ausente"):
        experiments.resume_configuration(extended, parent)


def test_campaign_preserves_failures_and_sequential_order(tmp_path, monkeypatch):
    config = recipe()
    config["runtime"]["storage_root"] = str(tmp_path)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(config))
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps({"purpose": "software_smoke_only", "operation": "smoke", "recipes": [path.name], "seeds": [42,43,44]}))
    calls = []
    def launch(operation, config, **kwargs):
        seed = config["training"]["seed"]
        calls.append(seed)
        if seed == 43:
            raise RuntimeError("CUDA out of memory")
        folder = tmp_path / str(seed)
        folder.mkdir()
        atomic_json(folder / "runtime.json", {"dependencies": {"framework": "1"}, "gpu": "fixture"})
        return {"run_path": str(folder), "run_id": str(seed), "compatibility": "same", "metrics": {"mAP50_95": .5 if seed == 42 else .7}}
    monkeypatch.setattr(experiments, "launch", launch)
    result = campaigns.campaign(campaign_path)
    assert calls == [42,43,44]
    assert result["counts"]["failed"] == 1 and not result["all_passed"]
    group = next(iter(result["groups"].values()))
    assert group["mean"] == pytest.approx(.6) and group["sample_std"] == pytest.approx(.141421356)
    again = campaigns.campaign(campaign_path, resume=result["campaign"])
    assert calls == [42,43,44] and again["counts"]["failed"] == 1
    state = load_json(Path(result["campaign"]) / "campaign.json")
    state["experiments"].append({"seed":45,"state":"pending"})
    atomic_json(Path(result["campaign"]) / "campaign.json", state, replace=True)
    assert not campaigns.summarize(result["campaign"])["complete"]


def test_summary_separates_actual_environment(tmp_path):
    experiments_list = []
    for seed, framework in [(42, "1"), (43, "2")]:
        folder = tmp_path / str(seed)
        folder.mkdir()
        atomic_json(folder / "runtime.json", {"dependencies": {"framework": framework}, "gpu": "fixture"})
        experiments_list.append({"seed": seed, "backend": "dfine", "state": "completed", "result": {
            "run_path": str(folder), "run_id": str(seed), "compatibility": "same", "metrics": {"mAP50_95": .5}}})
    atomic_json(tmp_path / "campaign.json", {"purpose": "software_smoke_only", "experiments": experiments_list})
    summary = campaigns.summarize(tmp_path)
    assert len(summary["groups"]) == 2
    assert all(g["sample_std"] is None for g in summary["groups"].values())


def test_full_image_measurement_extracts_tiles_from_original(project, tmp_path, monkeypatch):
    from conftest import save_yaml
    from ntv_ai.pipelines.experiment_worker import measure
    project["config"]["tiling"] = {"enabled": True, "size": 20, "overlap": 0, "border_policy": "clip"}
    save_yaml(project["prepare"], project["config"])
    dataset, manifest, records = prepared(project)
    selected = [r for r in records if r.image_id in manifest["view"]["image_ids"] and r.split == "val"]
    assert all(r.metadata.get("transform") for r in selected)
    # Timing must not read the already materialized tiles; it includes extraction.
    for r in selected:
        (Path(manifest["sources"][r.dataset_id]["root"]) / r.path).unlink()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    seen = []
    def predictor(image):
        seen.append(image.size)
        return [], 0.001
    result = measure(predictor, selected, manifest["sources"], {"warmup": 1, "repeats": 1, "tile_iou": .5})
    assert len(result["measurements"]) == 1
    assert len(seen) == 2 * len(selected)
    assert result["measurements"][0]["model_seconds"] == pytest.approx(len(selected) * .001)


def test_gpu_lock_is_shared_between_launchers():
    import subprocess
    from ntv_ai.models.runtime import gpu_slot
    code = """
import fcntl, os, tempfile
from pathlib import Path
path = Path(tempfile.gettempdir()) / f'ntv-ai-{os.getuid()}-gpu-987654.lock'
with path.open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
    raise SystemExit(1)
"""
    with gpu_slot(987654):
        subprocess.run([sys.executable, "-c", code], check=True)
