from pathlib import Path

import pytest

from ntv_ai.experiments.recipes import resolve_recipe


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["yolo26-s", "dfine-s", "rfdetr-nano"])
def test_versioned_recipes_share_smoke_budget(name):
    config = resolve_recipe(ROOT / "configs/experiments" / f"{name}.yaml")
    assert config["training"]["epochs"] == 1
    assert config["training"]["batch"] == 1
    assert config["training"]["seed"] == 42
    assert config["training"]["workers"] == 0
    assert config["protocol"]["resolution"] == 320
    assert config["purpose"] == "software_smoke_only"
    for size in (640,1280):
        profile = resolve_recipe(ROOT / "configs/experiments" / f"{name}-memory-{size}.yaml")
        assert profile["protocol"]["resolution"] == size
        assert profile["training"]["amp"] and profile["training"]["batch"] == 1
