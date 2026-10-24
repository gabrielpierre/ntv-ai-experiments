from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import sys
import time

import pytest

from ntv_ai.domain.records import Annotation, Geometry, ImageRecord
from ntv_ai.experiments.workshop_metrics import parse_qwen, operating_metrics, choose_threshold
from ntv_ai.experiments.workshop_prompts import CLASSES, grounding_text, bank
from ntv_ai.pipelines.benchmark_data import descriptor
from ntv_ai.pipelines.workshop_data import compose
from ntv_ai.storage import atomic_json, write_snapshot, load_json


def row(dataset, split, classes, number=0):
    key=f'{dataset}-{split}-{number}'
    ann=Annotation(key+'-box', classes[0], Geometry('bbox',[1,2,7,8]), {}, label=classes[0], mapping_status='mapped')
    return ImageRecord(key,dataset,key+'.png',10,10,key,key,[ann],dict.fromkeys(classes,'exhaustive'),
                       key,split,metadata={'grouping_reviewed':True})


def entry(tmp, dataset, classes):
    rows=[row(dataset,s,classes) for s in ('train','val','test')]
    source={dataset:{'root':str(tmp),'version':dataset,'evidence':{}}}
    view={'id':dataset,'classes':classes,'image_ids':[r.image_id for r in rows],
          'evaluation_image_ids':[r.image_id for r in rows], 'eligible':True}
    return descriptor(write_snapshot(tmp/dataset,rows,source,{},[],view=view))


def test_mixture_keeps_only_declared_training_and_target_validation(tmp_path):
    classes=['isolador','espacador','amortecedor']
    entries={name:entry(tmp_path,name,classes) for name in ('ptl_ai','stn_plad','ntv_ai')}
    rows,sources,originals,samples=compose(entries,['ptl_ai','stn_plad'],classes)
    assert {(r.dataset_id,r.split) for r in rows}=={('ptl_ai','train'),('stn_plad','train'),('ntv_ai','val')}
    assert len(originals)==len(samples)==3
    assert all(r.annotation_scope==dict.fromkeys(classes,'exhaustive') for r in rows)
    original=json.loads((tmp_path/'ptl_ai'/'images.jsonl').read_text().splitlines()[0])
    assert 'test' in (tmp_path/'ptl_ai'/'images.jsonl').read_text()


def test_mixture_rejects_missing_coverage(tmp_path):
    classes=['isolador','espacador','amortecedor']
    entries={name:entry(tmp_path,name,classes if name!='ptl_ai' else ['isolador'])
             for name in ('ptl_ai','stn_plad','ntv_ai')}
    with pytest.raises(ValueError,match='coverage'):
        compose(entries,['ptl_ai','stn_plad'],classes)


def test_mixture_rejects_leakage(monkeypatch):
    import ntv_ai.pipelines.workshop_data as mod
    classes=['isolador']
    datasets={}
    for ds in ('ptl_ai','ntv_ai'):
        rows=[row(ds,s,classes) for s in ('train','val','test')]
        datasets[ds]=({'view':{'classes':classes,'image_ids':[r.image_id for r in rows],
                     'evaluation_image_ids':[r.image_id for r in rows]},'sources':{}},rows)
    datasets['ntv_ai'][1][2].pixel_sha256=datasets['ptl_ai'][1][0].pixel_sha256
    monkeypatch.setattr(mod,'load_dataset',lambda name:datasets[name])
    with pytest.raises(ValueError,match='Vazamento'):
        compose({'ptl_ai':'ptl_ai','ntv_ai':'ntv_ai'},['ptl_ai'],classes)


def test_qwen_score_uses_only_class_and_coordinate_tokens():
    text='{"detections":[{"class_id":"isolador","bbox_xyxy":[100,200,700,800]}]}'
    spans=[(i,i+1) for i in range(len(text))]
    label_start=text.index('isolador')
    probabilities=[math.log(.8) if text[i].isdigit() or label_start<=i<label_start+8 else math.log(.01)
                   for i in range(len(text))]
    # Low-probability JSON punctuation must not lower the object's score.
    dets,errors=parse_qwen(text,spans,probabilities,['isolador'],(200,100))
    assert not errors
    assert dets[0].score==pytest.approx(.8)
    assert dets[0].bbox_xyxy==[20,20,140,80]
    text2='{"detections":[]}'
    assert parse_qwen(text2,[],[],['isolador'],(200,100))==([],[])


@pytest.mark.parametrize('value',[
    'not JSON', '{"detections":', '{"detections":[{"class_id":"unknown","bbox_xyxy":[0,0,2,2]}]}',
    '{"detections":[{"class_id":"isolador","bbox_xyxy":[0,0,1001,2]}]}',
    '{"detections":[{"class_id":"isolador","bbox_xyxy":[2,0,1,2]}]}',
    '{"detections":[{"class_id":"isolador","class_id":"torre","bbox_xyxy":[0,0,1,2]}]}',
    '{"detections":[{"class_id":"isolador","bbox_xyxy":[0,0,NaN,2]}]}'
])
def test_bad_model_output_is_recorded(value):
    detections,errors=parse_qwen(value,[(i,i+1) for i in range(len(value))],[-.1]*len(value),['isolador'],(20,20))
    assert not detections and errors


def test_duplicate_detection_is_false_positive_and_empty_result_is_false_negative():
    r=row('a','test',['isolador'])
    pred={'image_id':r.image_id,'detections':[{'class_id':'isolador','bbox_xyxy':[1,2,7,8],'score':.9},
                                             {'class_id':'isolador','bbox_xyxy':[1,2,7,8],'score':.2}]}
    metrics=operating_metrics([r],[pred],['isolador'],.001)
    assert metrics['micro']['tp']==1 and metrics['micro']['fp']==1
    best=choose_threshold([r],[pred],['isolador'])
    assert best['micro']['f1']==1 and best['threshold']==.3
    pred['detections']=[]
    assert operating_metrics([r],[pred],['isolador'],.001)['micro']['fn']==1


def test_prompt_spans_are_exact_and_all_classes_are_preserved():
    frozen=bank()
    assert frozen['classes']==CLASSES
    for prompt in range(4):
        text,spans=grounding_text(prompt)
        assert [text[a:b] for a,b in spans]==frozen['families'][str(prompt)]['phrases']
        assert len(spans)==7


def test_runner_budget_prevents_launch_and_keeps_pending_visible(tmp_path,monkeypatch):
    from ntv_ai.experiments.workshop_runner import Driver
    cfg=tmp_path/'configs/benchmark/workshop-campaign.yaml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f'workspace: {tmp_path.as_posix()}/out\ndeadline: "2026-10-15T23:59:00-03:00"\nreport_reserve_seconds: 3600\n')
    d=Driver(cfg)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**k:pytest.fail('must not launch'))
    assert d.node('train/42/full640/test',{'kind':'train'},time.time()-1) is None
    assert d.state['nodes']['train/42/full640/test']['state']=='budget_exhausted'


def test_runner_completed_node_is_not_repeated_and_hash_changes_block(tmp_path,monkeypatch):
    from ntv_ai.experiments.workshop_runner import Driver
    cfg=tmp_path/'configs/benchmark/workshop-campaign.yaml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f'workspace: {tmp_path.as_posix()}/out\ndeadline: "2026-10-15T23:59:00-03:00"\nreport_reserve_seconds: 3600\n')
    d=Driver(cfg)
    out=tmp_path/'out/jobs/smoke/example'
    atomic_json(out/'result.json',{'ok':True})
    monkeypatch.setattr('subprocess.Popen',lambda *a,**k:pytest.fail('must not launch'))
    assert d.node('smoke/example',{'kind':'train'},time.time()+60)=={'ok':True}
    assert d.node('smoke/example',{'kind':'train'},time.time()+60)=={'ok':True}
    atomic_json(out/'result.json',{'ok':False},replace=True)
    with pytest.raises(ValueError,match='changed'):
        d.node('smoke/example',{'kind':'train'},time.time()+60)


def test_deadline_terminates_real_worker_group(tmp_path,monkeypatch):
    import subprocess
    from ntv_ai.experiments.workshop_runner import Driver, process_alive
    cfg=tmp_path/'configs/benchmark/workshop-campaign.yaml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f'workspace: {tmp_path.as_posix()}/out\ndeadline: "2026-10-15T23:59:00-03:00"\nreport_reserve_seconds: 3600\n')
    d=Driver(cfg)
    original=subprocess.Popen
    monkeypatch.setattr(subprocess,'Popen',lambda cmd,**kw:original([sys.executable,'-c','import time; time.sleep(90)'],**kw))
    assert d.node('smoke/deadline',{'kind':'train'},time.time()+.5) is None
    item=d.state['nodes']['smoke/deadline']
    assert item['state']=='budget_exhausted'
    assert not process_alive(item['process'])


def test_live_worker_cannot_be_duplicated(tmp_path,monkeypatch):
    from ntv_ai.experiments.workshop_runner import Driver
    cfg=tmp_path/'configs/benchmark/workshop-campaign.yaml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f'workspace: {tmp_path.as_posix()}/out\ndeadline: "2026-10-15T23:59:00-03:00"\nreport_reserve_seconds: 3600\n')
    d=Driver(cfg)
    assert d.node('smoke/live',{'kind':'train'},time.time()-1) is None
    d.state['nodes']['smoke/live']['process']={'pid':1}
    monkeypatch.setattr('ntv_ai.experiments.workshop_runner.process_alive',lambda _:True)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(RuntimeError,match='still active'):
        d.node('smoke/live',{'kind':'train'},time.time()+60)
