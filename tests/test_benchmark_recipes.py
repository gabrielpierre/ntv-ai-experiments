from pathlib import Path

import pytest

from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.control import DEFAULT_CONTROL
from ntv_ai.experiments.recipes import resolve_recipe


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name',['yolo26-s','dfine-s','rfdetr-nano'])
@pytest.mark.parametrize('protocol,resolution',[('full640',640),('tiles1280',1280)])
def test_controlled_benchmark_recipe(name,protocol,resolution):
    cfg=resolve_recipe(ROOT/f'configs/benchmark/{name}-{protocol}.yaml')
    t=cfg['training']
    assert cfg['purpose']=='exploratory'
    assert cfg['dataset']=={'catalog_required':True}
    assert cfg['protocol']['resolution']==resolution and cfg['protocol']['weight_policy']=='ema'
    assert t['control']==DEFAULT_CONTROL
    assert t['epochs']==10000 and t['batch']==1 and t['workers']==0 and t['amp']
    if name=='yolo26-s':
        assert t['native']['optimizer']=='AdamW' and t['native']['loss_horizon']==100
        assert all(t['native'][k]==0 for k in ('mosaic','fliplr','scale','translate'))
    elif name=='dfine-s':
        assert t['native']['augment'] is False


def test_pilot_has_no_test_nodes_and_long_matrix_has_all_bases():
    pilot=read_config(ROOT/'configs/benchmark/pilot.yaml')
    long=read_config(ROOT/'configs/benchmark/campaign.yaml')
    assert pilot['epochs']==5 and pilot['seeds']==[42]
    assert pilot['targets']==[] and pilot['origins']==['ntv_ai']
    assert pilot['operation']=='smoke'
    assert long['seeds']==[42,43,44]
    assert len(long['origins'])==6 and long['origins']==long['targets']
    assert pilot['recipes']==long['recipes']
