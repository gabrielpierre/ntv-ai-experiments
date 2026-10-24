from pathlib import Path
from types import SimpleNamespace

import pytest

from ntv_ai.experiments.recipes import resolve_recipe, validate_recipe

from ntv_ai.models.backends import rfdetr

ROOT = Path(__file__).resolve().parents[1]


def test_rfdetr_native_options_never_evaluate_test(tmp_path):
    config = resolve_recipe(ROOT / "configs/experiments/rfdetr-nano.yaml")
    config["classes"] = ["tower"]
    opts = rfdetr.options(config, tmp_path / "export", tmp_path)
    assert opts["run_test"] is False
    assert opts["multi_scale"] is False and opts["scale_jitter"] is False
    assert opts["batch_size"] == opts["grad_accum_steps"] == opts["eval_batch_size"] == 1
    assert opts["eval_max_dets"] == config["protocol"]["max_detections"]


def test_background_cannot_take_foreground_topk_slots():
    import numpy as np
    outputs = {'pred_logits': np.array([[[.1, .2, 10.0]]]), 'pred_boxes': 'unchanged'}
    foreground = rfdetr.foreground_outputs(outputs, ['torre', 'isolador'])
    assert foreground['pred_logits'].shape == (1, 1, 2)
    assert foreground['pred_logits'].argmax() == 1
    assert foreground['pred_boxes'] == 'unchanged'
    assert outputs['pred_logits'].shape == (1, 1, 3)
    with pytest.raises(ValueError, match='Cabeça'):
        rfdetr.foreground_outputs(outputs, ['torre'])
