import os
from pathlib import Path

from PIL import Image
import pytest

from ntv_ai.pipelines.model_packages import predict_package
from ntv_ai.storage import load_json


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("NTV_GPU_TESTS") != "1" or not os.environ.get("NTV_MODEL_PACKAGE"),
                    reason="Requer NTV_GPU_TESTS=1, NTV_MODEL_PACKAGE e ambientes Conda")
@pytest.mark.parametrize("model", ["yolo26-s", "dfine-s", "rfdetr-nano"])
@pytest.mark.parametrize("protocol", ["full640", "tiles1280"])
def test_package_native_inference_without_training_dataset(model, protocol, tmp_path):
    import json
    source = tmp_path / "input.png"
    Image.new("RGB", (1400, 1300), (127, 127, 127)).save(source)
    root = Path(__file__).resolve().parents[1]
    result = predict_package(os.environ["NTV_MODEL_PACKAGE"], model, protocol, 42, source,
                             storage_root=root, runs_root=tmp_path / "runs")
    assert result["images"] == 1 and result["protocol"] == protocol
    path = Path(result["predictions"]["path"])
    assert path.is_file() and "sha256" not in result["predictions"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "processed"
    for detection in rows[0]["detections"]:
        assert detection["class_id"] in result["classes"]
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        assert 0 <= x1 < x2 <= 1400 and 0 <= y1 < y2 <= 1300
    assert result["timings"][0]["samples"] == (1 if protocol == "full640" else 4)
    assert load_json(Path(result["run_path"]) / "status.json")["state"] == "completed"
