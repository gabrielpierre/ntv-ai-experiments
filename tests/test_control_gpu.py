"""Opt-in native controller integration; COCO8 only, never the held-out NTV test."""
import os
from copy import deepcopy
from pathlib import Path

import pytest

from ntv_ai.experiments.control import DEFAULT_CONTROL
from ntv_ai.experiments.recipes import resolve_recipe
from ntv_ai.pipelines.experiments import launch
from ntv_ai.storage import load_json


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get('NTV_GPU_CONTROL') != '1',
             reason='Controle GPU opt-in; requer ambientes e COCO8 adquiridos')]


def config(name):
    root=Path(__file__).resolve().parents[1]
    cfg=resolve_recipe(root/f'configs/experiments/{name}.yaml')
    cfg['training']['control']=deepcopy(DEFAULT_CONTROL)
    if name=='yolo26-s': cfg['training']['native']['loss_horizon']=100
    return cfg


@pytest.mark.parametrize('name',['yolo26-s','dfine-s','rfdetr-nano'])
def test_common_early_stop_native(name, tmp_path):
    cfg=config(name)
    cfg['training']['epochs']=5
    cfg['training']['control'].update(warmup_epochs=0, patience=2, min_delta=1.0, plateau_patience=1)
    result=launch('smoke',cfg,runs_root=tmp_path/'runs')
    assert result['last_epoch']==2 and result['stop_reason']=='early_stopping'
    assert result['control_state']['bad_epochs']==2 and result['control_state']['lr_factor']==.25
    assert result['reevaluation_equal']
    backend=Path(result['run_path'])/'backend'
    assert len(list(backend.glob('resume-*.control.json')))==2
    with pytest.raises(ValueError,match='early stopping'):
        launch('train',cfg,resume=result['run_path'],runs_root=tmp_path/'runs')


@pytest.mark.parametrize('name',['yolo26-s','dfine-s','rfdetr-nano'])
def test_native_resume_preserves_control_and_epochs(name, tmp_path):
    cfg=config(name); cfg['training']['epochs']=2
    first=launch('train',cfg,runs_root=tmp_path/'runs')
    cfg['training']['epochs']=4
    resumed=launch('smoke',cfg,resume=first['run_path'],runs_root=tmp_path/'runs')
    assert resumed['last_epoch']==3
    assert resumed['control_state']['steps'] > first['control_state']['steps']
    assert resumed['control_state']['warmup_steps']==first['control_state']['warmup_steps']
    assert resumed['control_state']['base_lrs']==first['control_state']['base_lrs']
    assert resumed['reevaluation_equal']
    history=load_json(Path(resumed['run_path'])/'selection/selection.json')['history']
    assert [r['epoch'] for r in history]==[0,1,2,3]
    assert load_json(Path(first['run_path'])/'status.json')['state']=='completed'
