from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import threading
import time

import pytest

from ntv_ai.data.stratified import split_targets, stratify_groups
from ntv_ai.domain.records import Annotation, Geometry, ImageRecord
from ntv_ai.storage import atomic_json, load_json


def rows(n=30):
    result = []
    for i in range(n):
        r = ImageRecord(str(i), 'fixture', f'{i}.jpg', width=100, height=100, sha256=f'h{i}', group_id=f'g{i}')
        r.metadata['material'] = ['vidro', 'ceramica', 'polimero'][i % 3]
        r.annotations = [Annotation(str(i), 'a', Geometry('bbox', [1, 1, 5, 5]), {}, label='a', mapping_status='mapped')]
        if i % 4 == 0:
            r.annotations.append(Annotation(f'b{i}', 'b', Geometry('bbox', [7, 7, 9, 9]), {}, label='b', mapping_status='mapped'))
        result.append(r)
    return result


def test_targets_match_useful_ntv_ratio():
    assert split_targets(100) == {'train': 65, 'val': 18, 'test': 17}
    assert split_targets(97) == {'train': 63, 'val': 17, 'test': 17}
    assert split_targets(88) == {'train': 58, 'val': 15, 'test': 15}


def test_stratification_keeps_groups_classes_and_is_reproducible():
    original = rows()
    original[0].group_id = original[1].group_id
    first, second = deepcopy(original), deepcopy(original)
    a = stratify_groups(first, ['a', 'b'])
    b = stratify_groups(second, ['a', 'b'])
    assert a == b
    assert [r.split for r in first] == [r.split for r in second]
    assert first[0].split == first[1].split
    assert a['actual'] == a['targets']
    for split in ('train', 'val', 'test'):
        assert {x.label for r in first if r.split == split for x in r.annotations} == {'a', 'b'}
    assert all(r.split is None for r in original)


def test_stratification_fails_if_class_cannot_cover_splits():
    data = rows(8)
    with pytest.raises(ValueError, match='fewer than three'):
        stratify_groups(data, ['a', 'b'])


def test_scheduler_bounds_jobs_and_obeys_dependencies(tmp_path, monkeypatch):
    from ntv_ai.experiments import scheduler, matrix, cross
    from ntv_ai.models import runtime
    from ntv_ai.pipelines import experiments
    config = {'runtime': {'storage_root': str(tmp_path), 'device': 0}}
    nodes = []
    for i in range(4):
        common = {'backend': 'fake', 'variant': 's', 'protocol': 'full640', 'seed': i}
        nodes.append({'id': f't{i}', 'kind': 'train', 'state': 'pending', 'config': config, **common})
        nodes.append({'id': f'e{i}', 'kind': 'evaluate', 'state': 'pending', 'depends_on': f't{i}', 'target_dataset': {}, **common})
    state = {'experiments': nodes, 'execution': {'max_concurrent_jobs': 2, 'memory_mib': {'fake-s/full640': 2000}},
             'operation': 'train', 'purpose': 'exploratory', 'dataset_readiness': {}}
    monkeypatch.setattr(matrix, 'build_plan', lambda *a: deepcopy(state))
    monkeypatch.setattr(matrix, 'summarize_matrix', lambda folder: {
        'counts': dict(Counter(x['state'] for x in load_json(folder / 'campaign.json')['experiments'])), 'complete': True})
    monkeypatch.setattr(scheduler, 'gpu_memory', lambda device: (12000, 500))
    @contextmanager
    def lease(device):
        yield 123
    monkeypatch.setattr(runtime, 'gpu_slot', lease)
    active = 0
    maximum = 0
    lock = threading.Lock()
    def work(*args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        assert kwargs['execution']['lease_fd'] == 123
        time.sleep(.05)
        with lock:
            active -= 1
        return {'run_path': str(tmp_path / 'trained')}
    monkeypatch.setattr(experiments, 'launch', work)
    monkeypatch.setattr(cross, 'cross_predict', work)
    cfg = tmp_path / 'config.json'
    atomic_json(cfg, {'purpose': 'exploratory'})
    result = scheduler.concurrent_campaign(cfg, 'catalog', runs_root=tmp_path / 'runs')
    assert result['counts'] == {'completed': 8}
    assert maximum == 2


def test_inherited_gpu_lease_blocks_other_processes(tmp_path):
    import subprocess
    import sys
    from ntv_ai.models.runtime import gpu_slot
    with gpu_slot(98765) as fd:
        child = subprocess.run([sys.executable, '-c',
            f'from ntv_ai.models.runtime import gpu_slot\nwith gpu_slot(98765, {fd}): print("inherited")'],
            pass_fds=(fd,), capture_output=True, text=True, check=True)
        assert 'inherited' in child.stdout
        other = subprocess.run([sys.executable, '-c',
            'import fcntl,os,tempfile\n'
            'p=f"{tempfile.gettempdir()}/ntv-ai-{os.getuid()}-gpu-98765.lock"\n'
            'with open(p,"a") as f: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'],
            capture_output=True, text=True)
        assert other.returncode != 0 and 'BlockingIOError' in other.stderr


def test_parallel_tiles_match_serial_pixels_and_records(tmp_path):
    from PIL import Image
    from ntv_ai.pipelines.prepare import materialize_tiles
    data = rows(3)
    for r in data:
        Image.new('RGB', (100, 100), color=(int(r.image_id)*40, 2, 3)).save(tmp_path / r.path)
    sources = {'fixture': {'root': str(tmp_path)}}
    config = {'size': 64, 'overlap': .25, 'border_policy': 'clip'}
    a = materialize_tiles(data, deepcopy(sources), ['a'], config, tmp_path / 'serial', workers=1)
    b = materialize_tiles(data, deepcopy(sources), ['a'], config, tmp_path / 'parallel', workers=3)
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


def test_scheduler_oom_retries_with_native_progress_and_lower_concurrency(tmp_path, monkeypatch):
    from ntv_ai.experiments import scheduler, matrix
    from ntv_ai.models import runtime
    from ntv_ai.pipelines import experiments
    folder = tmp_path / 'campaign'
    folder.mkdir()
    node = {'id': 'a', 'kind': 'train', 'state': 'pending', 'backend': 'fake', 'variant': 's',
            'protocol': 'full640', 'seed': 42, 'config': {'runtime': {'device': 0}}}
    state = {'experiments': [node], 'operation': 'train', 'execution':
             {'max_concurrent_jobs': 2, 'memory_mib': {'fake-s/full640': 2000}}}
    atomic_json(folder / 'campaign.json', state)
    monkeypatch.setattr(scheduler, 'gpu_memory', lambda device: (12000, 500))
    @contextmanager
    def lease(device):
        yield 123
    monkeypatch.setattr(runtime, 'gpu_slot', lease)
    monkeypatch.setattr(matrix, 'summarize_matrix', lambda folder: {'counts': {}, 'complete': True})
    attempts = []
    def launch(*args, **kwargs):
        attempt = tmp_path / f'attempt-{len(attempts)}'
        attempt.mkdir()
        kwargs['on_created'](attempt)
        if not attempts:
            attempts.append(attempt)
            atomic_json(attempt / 'failure.json', {'oom': True})
            atomic_json(attempt / 'progress.json', {'last_epoch': 3})
            raise RuntimeError('CUDA out of memory')
        assert kwargs['resume'] == str(attempts[0])
        assert kwargs['execution']['max_concurrent_jobs'] == 1
        attempts.append(attempt)
        return {'run_path': str(attempt)}
    monkeypatch.setattr(experiments, 'launch', launch)
    scheduler.concurrent_campaign(None, resume=folder)
    final = load_json(folder / 'campaign.json')
    assert final['experiments'][0]['state'] == 'completed'
    assert final['experiments'][0]['oom_retries'] == 1
    scheduler.concurrent_campaign(None, resume=folder)
    assert len(attempts) == 2  # Completed training is never scheduled twice.


def test_reconciliation_refuses_live_worker_and_recovers_dead_worker(tmp_path, monkeypatch):
    from ntv_ai.experiments import scheduler
    state = {'experiments': [{'id': 'a', 'state': 'running', 'attempts': [str(tmp_path)]}]}
    monkeypatch.setattr(scheduler, 'process_alive', lambda _: True)
    with pytest.raises(RuntimeError, match='still alive'):
        scheduler.reconcile(state)
    monkeypatch.setattr(scheduler, 'process_alive', lambda _: False)
    scheduler.reconcile(state)
    assert state['experiments'][0]['state'] == 'interrupted'


def test_matrix_keeps_shared_paths_and_restricts_targets(project, tmp_path):
    from test_cross_benchmark import pair
    from test_experiment_core import recipe
    from ntv_ai.experiments.matrix import build_plan
    source, target = pair(project, tmp_path)
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    datasets = {name: {'full640': {**entry, 'eligible': True, 'blockers': []}}
                for name, entry in [('source', source), ('target', target)]}
    atomic_json(checkout / 'catalog.json', {'purpose': 'exploratory', 'storage_root': str(checkout), 'datasets': datasets})
    cfg = recipe()
    cfg['purpose'] = 'exploratory'
    cfg['protocol']['resolution'] = 640
    atomic_json(checkout / 'recipe.json', cfg)
    atomic_json(checkout / 'matrix.json', {'schema_version': 2, 'purpose': 'exploratory', 'operation': 'train',
        'origins': ['source'], 'targets': ['unknown'], 'targets_by_origin': {'source': ['source', 'target']},
        'seeds': [42, 43, 44], 'recipes': {'full640': ['recipe.json']}})
    plan = build_plan(checkout / 'matrix.json', checkout / 'catalog.json', storage_root=checkout)
    assert Counter(n['kind'] for n in plan['experiments']) == {'train': 3, 'evaluate': 6}
    assert all(n['state'] == 'pending' for n in plan['experiments'])
    assert plan['experiments'][0]['config']['dataset']['manifest']['path'] == source['manifest']['path']
