"""Bounded, restartable campaign supervisor. Scientific recipes are unchanged."""
import concurrent.futures
import contextlib
import fcntl
import os
import subprocess
import threading
import time
from pathlib import Path

from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.recipes import compatibility
from ntv_ai.experiments.runs import Run, now
from ntv_ai.storage import atomic_json, load_json


def gpu_memory(device):
    result = subprocess.run(["nvidia-smi", "-i", str(device),
                             "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return tuple(int(v.strip()) for v in result.stdout.strip().split(","))


def process_alive(folder):
    ref = Path(folder) / "worker-process.json"
    if not ref.exists():
        return False
    identity = load_json(ref)
    stat = Path(f"/proc/{identity['pid']}/stat")
    try:
        values = stat.read_text().rsplit(")", 1)[1].split()
        return (values[0] != 'Z' and values[19] == identity['start_ticks'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot_id'])
    except FileNotFoundError:
        return False


def reconcile(state):
    for item in state["experiments"]:
        if item["state"] != "running":
            continue
        attempts = item.get("attempts", [])
        if not attempts:
            item["state"] = "interrupted"
            continue
        folder = Path(attempts[-1])
        if process_alive(folder):
            raise RuntimeError(f"Worker still alive for {item['id']}: {folder}")
        result = folder / "result.json"
        if result.exists():
            value = load_json(result)
            config = load_json(folder / 'config.json')
            value.update(run_id=folder.name, run_path=str(folder), compatibility=compatibility(config))
            atomic_json(result, value, replace=True)
            status = load_json(folder / 'status.json')
            status.update(state='completed', finished_at=now(), reconciled=True)
            atomic_json(folder / 'status.json', status, replace=True)
            item.update(state="completed", result=value, finished_at=now())
        else:
            item["state"] = "interrupted"


def recipe_key(item):
    return f"{item['backend']}-{item['variant']}/{item['protocol']}"


def concurrent_campaign(config_path, catalog=None, storage_root=None, runs_root=None, resume=None):
    from ntv_ai.experiments.matrix import build_plan, summarize_matrix
    from ntv_ai.experiments.cross import cross_predict
    from ntv_ai.pipelines.experiments import launch
    from ntv_ai.models.runtime import gpu_slot

    context = None
    if resume:
        folder = Path(resume).resolve()
        state = load_json(folder / 'campaign.json')
    else:
        state = build_plan(config_path, catalog, storage_root)
        source = next(x for x in state['experiments'] if x['kind'] == 'train')['config']
        context = Run(Path(runs_root or Path(source['runtime']['storage_root']) / 'runs'),
                      'matrix_campaign', read_config(config_path))
        folder = context.path
        atomic_json(folder / 'campaign.json', state)
    execution = state['execution']
    capacity = execution.get('max_concurrent_jobs', 1)
    if type(capacity) is not int or not 1 <= capacity <= 3:
        raise ValueError('Concurrency must be between one and three jobs')
    estimates = execution.get('memory_mib', {})
    if capacity > 1 and any(recipe_key(x) not in estimates for x in state['experiments']):
        raise ValueError('Concurrent execution requires a measured memory bound for every recipe')
    device = next(x for x in state['experiments'] if x['kind'] == 'train')['config']['runtime']['device']
    index = {x['id']: x for x in state['experiments']}
    mutex = threading.RLock()

    def save():
        atomic_json(folder / 'campaign.json', state, replace=True)

    def execute(item, lease):
        def created(path):
            with mutex:
                item.setdefault('attempts', []).append(str(path))
                save()
        settings = {**execution, 'lease_fd': lease, 'admitted_at': now(), 'mode': 'supervised_pool'}
        if item['kind'] == 'train':
            parent = None
            if item.get('resume_from') and (Path(item['resume_from']) / 'progress.json').exists():
                parent = item['resume_from']
            return launch(state['operation'], item['config'], runs_root=runs_root or folder.parent,
                          resume=parent, on_created=created, execution=settings)
        return cross_predict(index[item['depends_on']]['result']['run_path'], item['target_dataset'],
                             runs_root=runs_root or folder.parent, execution=settings, on_created=created)

    try:
        with (folder / 'supervisor.lock').open('a') as supervisor:
            fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reconcile(state)
            save()
            with gpu_slot(device) as lease, concurrent.futures.ThreadPoolExecutor(max_workers=capacity) as pool:
                total, baseline = gpu_memory(device)
                reserve = execution.get('gpu_reserve_mib', 2048)
                usable = total - baseline - reserve
                active = {}
                while True:
                    with mutex:
                        for future in list(active):
                            if not future.done():
                                continue
                            item, memory = active.pop(future)
                            try:
                                result = future.result()
                                item.update(state=result.get('state', 'completed'), result=result, finished_at=now())
                            except Exception as exc:
                                attempts = item.get('attempts', [])
                                failure = Path(attempts[-1]) / 'failure.json' if attempts else None
                                oom = failure and failure.exists() and load_json(failure).get('oom')
                                if oom and item.get('oom_retries', 0) < 2:
                                    capacity = 1
                                    state['execution']['max_concurrent_jobs'] = 1
                                    item.update(state='interrupted', oom_retries=item.get('oom_retries', 0) + 1,
                                                error=str(exc), retry_after_drain=True)
                                else:
                                    item.update(state='failed', error=str(exc), finished_at=now())
                            save()
                        for item in state['experiments']:
                            if item['state'] not in ('pending', 'interrupted', 'blocked'):
                                continue
                            if item['state'] == 'blocked' and item.get('reason') != 'training_dependency_not_completed':
                                continue
                            if item['kind'] == 'evaluate':
                                dependency = index[item['depends_on']]
                                if dependency['state'] in ('failed', 'blocked'):
                                    item.update(state='blocked', reason='training_dependency_not_completed')
                                    continue
                                if dependency['state'] != 'completed':
                                    continue
                            if item.get('retry_after_drain') and active:
                                continue
                            serial = recipe_key(item) in execution.get('serial_recipes', [])
                            if len(active) >= capacity or (serial and active):
                                continue
                            if any(recipe_key(x[0]) in execution.get('serial_recipes', []) for x in active.values()):
                                continue
                            memory = estimates.get(recipe_key(item), usable)
                            if memory > usable:
                                raise RuntimeError(f"Insufficient memory reserve for {item['id']}")
                            if sum(x[1] for x in active.values()) + memory > usable:
                                continue
                            _, used = gpu_memory(device)
                            if total - used < reserve + min(memory, 1024):
                                continue
                            if item['state'] == 'interrupted' and item.get('attempts'):
                                item['resume_from'] = item['attempts'][-1]
                            item.pop('reason', None)
                            item.pop('retry_after_drain', None)
                            item.update(state='running', started_at=now())
                            save()
                            active[pool.submit(execute, item, lease)] = (item, memory)
                        if not active:
                            pending = [x for x in state['experiments'] if x['state'] in ('pending', 'interrupted')]
                            if not pending:
                                break
                            raise RuntimeError('Pending jobs could not be admitted; inspect resources/dependencies')
                    time.sleep(.25)
                save()
        report = summarize_matrix(folder)
        if context:
            context.__exit__(None, None, None)
        return {'campaign': str(folder), 'counts': report['counts'], 'complete': report['complete']}
    except BaseException as exc:
        if context:
            context.__exit__(type(exc), exc, exc.__traceback__)
        raise
