from copy import deepcopy
from pathlib import Path

import pytest

from ntv_ai.experiments.cross import check_leakage, evaluation_data, pair_ready
from ntv_ai.experiments import matrix
from ntv_ai.domain.records import Detection
from ntv_ai.pipelines.benchmark_data import descriptor
from ntv_ai.pipelines.cross_worker import work_cross
from ntv_ai.storage import atomic_json, load_json, write_snapshot
from test_experiment_core import prepared, recipe


def snapshot(folder, manifest, records, classes):
    records = deepcopy(records)
    for row in records:
        row.annotation_scope = dict.fromkeys(classes, 'exhaustive')
    view = {**manifest['view'], 'classes': classes, 'eligible': True,
            'image_ids': [r.image_id for r in records],
            'evaluation_image_ids': [r.image_id for r in records]}
    path = write_snapshot(folder, records, manifest['sources'], {'test': True}, [], view=view)
    return descriptor(path)


def pair(project, tmp_path):
    _, manifest, records = prepared(project)
    source = snapshot(tmp_path/'source', manifest, records, ['torre', 'isolador'])
    target_rows = deepcopy([r for r in records if r.split == 'test'])
    negative = deepcopy(target_rows[0])
    negative.image_id += '-negative'
    negative.annotations = []
    target_rows.append(negative)
    return source, snapshot(tmp_path/'target', manifest, target_rows, ['isolador', 'amortecedor'])


def test_pair_accepts_test_only_target_and_preserves_negatives(project, tmp_path):
    source, target = pair(project, tmp_path)
    assert pair_ready(source, target) == {'state': 'pending', 'classes': ['isolador']}
    _, originals, samples = evaluation_data(target, ['isolador'])
    assert len(originals) == len(samples) == 2
    assert all(r.split == 'test' for r in samples)
    assert sum(not r.annotations for r in originals) == 1
    empty = {**source, 'classes': ['espacador']}
    assert pair_ready(empty, target)['state'] == 'not_applicable'


def test_target_training_images_are_not_required_for_evaluation(project, tmp_path):
    _, manifest, records = prepared(project)
    target = snapshot(tmp_path/'target',manifest,records,['isolador'])
    for row in records:
        if row.split != 'test': (project['raw']/row.path).unlink()
    _, originals, _ = evaluation_data(target,['isolador'])
    assert len(originals)==1 and originals[0].split=='test'


def test_cross_worker_keeps_source_head_and_filters_both_sides(project, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    source, target = pair(project, tmp_path)
    cfg = recipe()
    cfg.update(classes=source['classes'], dataset=source)
    calls = []
    class Model:
        def load_predictor(self, checkpoint, config):
            assert config['classes'] == ['torre', 'isolador']
            def predict(image):
                calls.append(image.size)
                found = [Detection('torre', [1, 1, 3, 3], .99)]
                if len(calls) == 1:
                    found.append(Detection('isolador', [4, 5, 14, 17], .98))
                return found, .001
            return predict
    out = tmp_path/'predicted'; out.mkdir()
    result = work_cross(Model(), cfg, {'target':target, 'checkpoint':str(tmp_path/'fake.pt'),
        'split':'test', 'source_run':'synthetic'}, out)
    assert result['classes'] == ['isolador'] and result['reevaluation_equal']
    assert result['metrics']['images'] == 2
    assert result['metrics']['mAP50_95'] == pytest.approx(1)
    import json
    predictions = [json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(predictions) == 2 and sum(not p['detections'] for p in predictions) == 1


@pytest.mark.parametrize('field', ['sha256', 'pixel_sha256', 'group_id'])
def test_leakage_includes_source_validation(project, field):
    _, _, records = prepared(project)
    origin = next(r for r in records if r.split == 'val')
    target = deepcopy(next(r for r in records if r.split == 'test'))
    setattr(target, field, getattr(origin, field))
    with pytest.raises(ValueError, match='Vazamento'):
        check_leakage(records, [target])


def test_target_partial_coverage_and_tampering_block_pair(project, tmp_path):
    source, target = pair(project, tmp_path)
    from ntv_ai.experiments.cross import load_dataset
    manifest, records = load_dataset(target)
    records[0].annotation_scope['isolador'] = 'partial'
    path = write_snapshot(tmp_path/'partial', records, manifest['sources'], {}, [], view=manifest['view'])
    assert pair_ready(source, descriptor(path))['state'] == 'blocked'
    target['manifest']['sha256'] = 'f'*64
    assert pair_ready(source, target)['state'] == 'blocked'


def test_pair_intersects_reviewed_supervision_without_approving_other_classes(project, tmp_path):
    from ntv_ai.experiments.cross import load_dataset
    source, target = pair(project, tmp_path)
    manifest, records = load_dataset(target)
    manifest['view']['classes'] = ['torre','isolador','amortecedor']
    manifest['view']['eligible'] = False
    for r in records: r.annotation_scope['torre']='unknown'
    p=write_snapshot(tmp_path/'mixed-coverage',records,manifest['sources'],{},[],view=manifest['view'])
    decision=pair_ready(source,descriptor(p))
    assert decision == {'state':'pending','classes':['isolador'], 'excluded_unverified_classes':['torre']}
    assert load_dataset(descriptor(p))[0]['view']['eligible'] is False


def test_matrix_dependencies_failures_and_blocked_sources(tmp_path, monkeypatch):
    from ntv_ai.pipelines import experiments
    cfg = recipe(); cfg['runtime']['storage_root'] = str(tmp_path)
    nodes = [
        {'id':'a', 'kind':'train', 'state':'pending', 'config':cfg},
        {'id':'a/test', 'kind':'evaluate', 'state':'pending', 'depends_on':'a', 'target_dataset':{}},
        {'id':'b', 'kind':'train', 'state':'pending', 'config':cfg},
        {'id':'b/test', 'kind':'evaluate', 'state':'pending', 'depends_on':'b', 'target_dataset':{}},
        {'id':'c', 'kind':'train', 'state':'blocked', 'reason':'curation_pending', 'config':cfg},
        {'id':'c/test', 'kind':'evaluate', 'state':'not_applicable'}]
    state = {'schema_version':2, 'purpose':'exploratory', 'operation':'train',
             'experiments':nodes, 'dataset_readiness':{}}
    monkeypatch.setattr(matrix, 'build_plan', lambda *a: deepcopy(state))
    path=tmp_path/'config.json'; atomic_json(path, {'purpose':'exploratory'})
    calls=[]
    def launch(*a, **k):
        calls.append('train')
        if len(calls)>1: raise RuntimeError('CUDA out of memory')
        return {'run_path':str(tmp_path/'trained')}
    def cross(*a, **k):
        calls.append('evaluate'); return {'state':'completed'}
    monkeypatch.setattr(experiments, 'launch', launch)
    monkeypatch.setattr(matrix, 'cross_predict', cross)
    result=matrix.matrix_campaign(path, catalog='synthetic')
    assert calls == ['train', 'evaluate', 'train']
    assert result['counts'] == {'completed':2, 'failed':1, 'blocked':2, 'not_applicable':1}
    assert not result['complete']
    matrix.matrix_campaign(path, resume=result['campaign'])
    assert len(calls)==3


def test_matrix_summary_does_not_mix_target_classes_or_recipe(tmp_path):
    nodes=[]
    for i, (seed, target, classes, compatibility) in enumerate([
        (42,'a',['isolador'],'same'), (43,'a',['isolador'],'same'),
        (42,'b',['isolador'],'same'), (42,'a',['torre'],'same'), (42,'a',['isolador'],'different')]):
        folder=tmp_path/str(i); folder.mkdir(); atomic_json(folder/'runtime.json', {'gpu':'fixture'})
        metrics={'mAP50_95':.5, 'AP50':.6, 'AP75':.4, 'AR50_95':.7,
                 'AP_by_size':{'small':.5, 'medium':None, 'large':None},
                 'per_class':{classes[0]:{'mAP50_95':.5}}, 'dataset_version':target, 'protocol':{'split':'test'}}
        nodes.append({'state':'completed', 'origin':'source', 'target':target, 'backend':'dfine',
            'protocol':'full640', 'seed':seed, 'result':{'metrics':metrics, 'compatibility':compatibility,
            'target_version':target, 'classes':classes, 'run_path':str(folder)}})
    atomic_json(tmp_path/'campaign.json', {'purpose':'exploratory','experiments':nodes,'dataset_readiness':{}})
    report=matrix.summarize_matrix(tmp_path)
    assert len(report['groups']) == 4
    double=next(g for g in report['groups'].values() if len(g['seeds'])==2)
    assert double['aggregate']['mAP50_95']['sample_std']==0


def test_interrupted_matrix_uses_linked_native_state(tmp_path, monkeypatch):
    from ntv_ai.pipelines import experiments
    parent=tmp_path/'parent'; parent.mkdir(); atomic_json(parent/'progress.json', {'last_epoch':1})
    folder=tmp_path/'campaign'; folder.mkdir()
    atomic_json(folder/'campaign.json', {'schema_version':2,'purpose':'exploratory', 'operation':'train',
        'dataset_readiness':{}, 'experiments':[
        {'id':'a','kind':'train','state':'interrupted','config':{},'attempts':[str(parent)]},
        {'id':'test','kind':'evaluate','state':'blocked','reason':'training_dependency_not_completed',
         'depends_on':'a','target_dataset':{}}]})
    def launch(*args, **kwargs):
        assert kwargs['resume'] == parent
        kwargs['on_created'](tmp_path/'child')
        return {'run_path':str(tmp_path/'child')}
    monkeypatch.setattr(experiments,'launch',launch)
    monkeypatch.setattr(matrix,'cross_predict',lambda *a,**k:{})
    result=matrix.matrix_campaign(None,resume=folder)
    assert result['counts']=={'completed':2}
    assert load_json(folder/'campaign.json')['experiments'][0]['attempts']==[str(parent),str(tmp_path/'child')]


def test_matrix_can_relocate_dataset_without_changing_identity(project, tmp_path):
    import shutil
    from ntv_ai.experiments.cross import load_dataset
    source, _ = pair(project,tmp_path)
    entry={**source, 'eligible':True, 'blockers':[]}
    catalog=tmp_path/'catalog.json'
    atomic_json(catalog, {'purpose':'exploratory', 'storage_root':str(tmp_path),
                         'datasets':{'source':{'full640':entry}}})
    cfg=recipe(); cfg['purpose']='exploratory'; cfg['protocol']['resolution']=640
    atomic_json(tmp_path/'recipe.json',cfg)
    atomic_json(tmp_path/'matrix.json', {'schema_version':2, 'purpose':'exploratory','operation':'train',
        'origins':['source'], 'targets':[], 'seeds':[42], 'recipes':{'full640':['recipe.json']}})
    moved=tmp_path/'other-disk'; moved.mkdir()
    for name in ['source','raw']: shutil.copytree(tmp_path/name,moved/name)
    plan=matrix.build_plan(tmp_path/'matrix.json',catalog,storage_root=moved)
    data=plan['experiments'][0]['config']['dataset']
    assert data['version']==source['version']
    manifest,_=load_dataset(data)
    assert manifest['sources']['fixture']['root']==str(moved/'raw')
    assert plan['experiments'][0]['state']=='pending'
