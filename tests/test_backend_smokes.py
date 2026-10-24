import os
from pathlib import Path

import pytest

from ntv_ai.pipelines.experiments import run_experiment
from ntv_ai.storage import load_json


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("NTV_GPU_TESTS") != "1", reason="GPU opt-in; requer acquire e ambientes preparados")
@pytest.mark.parametrize("recipe", ["yolo26-s", "dfine-s", "rfdetr-nano"])
def test_native_smoke_roundtrip(recipe, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = run_experiment("smoke", root / "configs/experiments" / f"{recipe}.yaml",
                            storage_root=root, runs_root=tmp_path / "runs")
    assert result["reevaluation_equal"]
    assert result["last_epoch"] == 0
    assert result["training"]["metadata"]["optimizer_steps"] > 0
    assert result["metrics"]["images"] == 4
    selection = load_json(Path(result["run_path"]) / "selection/selection.json")
    assert selection["metric"] == "mAP50_95" and selection["split"] == "val"
    assert len(selection["history"]) == 1
    assert result["resume"]["path"] != result["best"]["checkpoint"]["path"]
    measured = run_experiment("measure", root / "configs/experiments" / f"{recipe}.yaml",
                              storage_root=root, runs_root=tmp_path / "runs",
                              checkpoint=Path(result["best"]["checkpoint"]["path"]))
    latency = measured["latency"]
    assert latency["batch"] == 1 and latency["cuda_synchronized"]
    assert 0 < latency["mean_model_seconds"] < latency["mean_image_seconds"]


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("NTV_GPU_PROFILES") != "1", reason="Perfis GPU opt-in")
@pytest.mark.parametrize("recipe", ["yolo26-s", "dfine-s", "rfdetr-nano"])
@pytest.mark.parametrize("size", [640,1280])
def test_memory_forward_backward_update(recipe, size, tmp_path):
    root = Path(__file__).resolve().parents[1]
    try:
        result = run_experiment("profile", root / "configs/experiments" / f"{recipe}-memory-{size}.yaml",
                                storage_root=root, runs_root=tmp_path / "runs")
    except RuntimeError:
        failures = list((tmp_path / "runs").glob("*/failure.json"))
        assert len(failures) == 1 and load_json(failures[0])["oom"]
        assert load_json(failures[0].parent / "config.json")["protocol"]["resolution"] == size
    else:
        assert result["training"]["metadata"]["optimizer_steps"] > 0
        memory = load_json(Path(result["run_path"]) / "runtime.json")
        assert memory["peak_allocated_bytes"] > 0 and memory["amp"]
