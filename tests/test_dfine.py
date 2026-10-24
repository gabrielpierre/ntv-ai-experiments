from pathlib import Path
from types import SimpleNamespace

import pytest

from ntv_ai.experiments.recipes import resolve_recipe, validate_recipe

from ntv_ai.models.registry import specification

ROOT = Path(__file__).resolve().parents[1]


def test_dfine_requires_zero_based_export_and_pinned_implementation():
    config = resolve_recipe(ROOT / "configs/experiments/dfine-s.yaml")
    assert specification("dfine").formats == ("coco_zero_based",)
    from ntv_ai.models.backends.dfine import REVISION, native_config
    config["model"]["implementation"]["revision"] = "changed"
    with pytest.raises(ValueError, match="Revisão"):
        native_config(config, Path("unused"))
