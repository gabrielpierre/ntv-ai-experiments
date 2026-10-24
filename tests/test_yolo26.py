from pathlib import Path
from types import SimpleNamespace

import pytest

from ntv_ai.experiments.recipes import resolve_recipe, validate_recipe

from ntv_ai.models.backends import yolo26

ROOT = Path(__file__).resolve().parents[1]


def test_yolo_native_options_have_fixed_batch_and_no_auto_optimizer(tmp_path):
    config = resolve_recipe(ROOT / "configs/experiments/yolo26-s.yaml")
    opts = yolo26.options(config, tmp_path / "data.yaml", tmp_path)
    assert opts["nbs"] == opts["batch"] == 1
    assert opts["nms"] is False and opts["patience"] == 0
    config["training"]["native"]["optimizer"] = "auto"
    with pytest.raises(ValueError, match="auto"):
        validate_recipe(config)


@pytest.mark.parametrize("variant", ["n", "s", "m", "l", "x"])
def test_yolo_load_checks_architecture_and_variant(variant, monkeypatch, tmp_path):
    import sys
    model = SimpleNamespace(model=SimpleNamespace(yaml={"scale": variant, "yaml_file": "yolo26.yaml"},
                            model=[SimpleNamespace(one2one_cv2=object())]))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *args, **kwargs: model))
    monkeypatch.setattr(yolo26, "version", lambda name: "8.4.155")
    assert yolo26.load_model(tmp_path / "checkpoint.pt", variant) is model
    model.model.yaml["scale"] = "invalid"
    with pytest.raises(ValueError, match="variante"):
        yolo26.load_model(tmp_path / "checkpoint.pt", variant)
