"""Execution-only calibration on source train/val. Never touches held-out test."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import math
import time

from ntv_ai.experiments.cross import load_dataset
from ntv_ai.experiments.recipes import resolve_recipe
from ntv_ai.experiments.scheduler import gpu_memory
from ntv_ai.models.runtime import gpu_slot
from ntv_ai.pipelines.benchmark_data import descriptor
from ntv_ai.pipelines.experiments import launch
from ntv_ai.storage import atomic_json, load_json, write_snapshot


MODELS = ('yolo26-s', 'dfine-s', 'rfdetr-nano')
PROTOCOLS = ('full640', 'tiles1280')


def small_dataset(data, folder, train_images=4, val_images=2):
    manifest, records = load_dataset(data)
    original_ids = set(manifest['view']['evaluation_image_ids'])
    originals = [r for r in records if r.image_id in original_ids]
    selected = set()
    for split, count in [('train', train_images), ('val', val_images)]:
        candidates = sorted((r for r in originals if r.split == split), key=lambda r: r.image_id)
        selected.update(r.image_id for r in candidates[:count])
    keep = [r for r in records if r.image_id in selected or r.metadata.get('transform', {}).get('parent_image_id') in selected]
    view = deepcopy(manifest['view'])
    view['id'] += '-execution-calibration'
    view['image_ids'] = [i for i in view['image_ids'] if any(r.image_id == i for r in keep)]
    view['evaluation_image_ids'] = sorted(selected)
    path = write_snapshot(folder, keep, manifest['sources'], {'operation': 'execution_calibration', 'parent': manifest['version']}, [], view=view)
    return descriptor(path)


def calibrate(root, catalog_path, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_json(catalog_path)
    recipes = {}
    for protocol in PROTOCOLS:
        data = small_dataset(catalog['datasets']['stn_plad'][protocol], output / 'datasets' / protocol)
        for model in MODELS:
            config = resolve_recipe(root / f'configs/benchmark/{model}-{protocol}.yaml', root)
            config['dataset'] = data
            config['training']['epochs'] = 1
            config['runtime']['cpu_threads'] = 8
            recipes[f'{model}/{protocol}'] = config
    report = {'purpose': 'execution_calibration_only', 'uses_test': False, 'phases': {}, 'memory_mib': {}, 'serial_recipes': []}
    atomic_json(output / 'calibration.json', report)
    with gpu_slot(0) as lease:
        total, baseline = gpu_memory(0)
        budget = total - baseline - 2048
        phases = [(1, '1', False), (2, '2', False), (3, '3', False)]
        for concurrency, phase_name, enforce_serial in phases:
            serial_policy = set(report['serial_recipes']) if enforce_serial else set()
            phase = {'results': {}, 'failures': {}, 'peak_device_mib': baseline, 'max_active': 0}
            phase['serial_policy'] = sorted(serial_policy)
            phase['parity_failures'] = []
            start = time.monotonic()
            queue = list(recipes)
            active = {}
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                while queue or active:
                    _, used = gpu_memory(0)
                    phase['peak_device_mib'] = max(phase['peak_device_mib'], used)
                    for future in list(active):
                        key, bound, job_peak = active[future]
                        active[future] = (key, bound, max(job_peak, used - baseline))
                        if not future.done():
                            continue
                        key, bound, job_peak = active.pop(future)
                        try:
                            result = future.result()
                            runtime = load_json(Path(result['run_path']) / 'runtime.json')
                            phase['results'][key] = {
                                'run_path': result['run_path'], 'metric': result['metrics']['mAP50_95'],
                                'steps': result['training']['metadata']['optimizer_steps'],
                                'compatibility': result['compatibility'],
                                'optimizer': load_json(Path(result['run_path']) / 'optimizer.json'),
                                'duration_seconds': runtime['duration_seconds'],
                                'peak_reserved_mib': runtime['peak_reserved_bytes'] / 2**20,
                            }
                            if concurrency == 1:
                                report['memory_mib'][key] = math.ceil(max(job_peak * 1.10, runtime['peak_reserved_bytes'] / 2**20 + 256))
                        except Exception as exc:
                            phase['failures'][key] = str(exc)
                    for key in list(queue):
                        if len(active) >= concurrency:
                            break
                        if (key in serial_policy and active) or any(v[0] in serial_policy for v in active.values()):
                            continue
                        bound = report['memory_mib'].get(key, budget)
                        if sum(v[1] for v in active.values()) + bound > budget:
                            continue
                        queue.remove(key)
                        settings = {'lease_fd': lease, 'cpu_threads': 8, 'max_concurrent_jobs': concurrency,
                                    'mode': 'execution_calibration'}
                        future = pool.submit(launch, 'smoke', recipes[key], runs_root=output / f'phase-{phase_name}', execution=settings)
                        active[future] = (key, bound, 0)
                        phase['max_active'] = max(phase['max_active'], len(active))
                    if queue and not active:
                        raise RuntimeError('Calibration cannot maintain the required GPU memory reserve')
                    time.sleep(.25)
            phase['wall_seconds'] = time.monotonic() - start
            phase['reserve_respected'] = phase['peak_device_mib'] <= total - 2048
            report['phases'][phase_name] = phase
            atomic_json(output / 'calibration.json', report, replace=True)
            if concurrency == 1 and phase['failures']:
                raise RuntimeError(f"Sequential smoke failed: {phase['failures']}")
            if concurrency > 1:
                reference = report['phases']['1']['results']
                for key in recipes:
                    value = phase['results'].get(key)
                    valid = (value is not None and value['steps'] == reference[key]['steps'] and
                             value['compatibility'] == reference[key]['compatibility'] and
                             value['optimizer'] == reference[key]['optimizer'] and
                             abs(value['metric'] - reference[key]['metric']) <= 1e-6)
                    if not valid:
                        report['serial_recipes'].append(key)
                        phase['parity_failures'].append(key)
            report['phases'][phase_name] = phase
            atomic_json(output / 'calibration.json', report, replace=True)
            if phase_name == '3' and 0 < len(set(report['serial_recipes'])) < len(recipes):
                # Measure the actual policy that production will use. Timings of
                # an unsafe concurrent mix cannot justify a throughput claim.
                phases.extend([(2, 'verified-2', True), (3, 'verified-3', True)])
    serial = sorted(set(report['serial_recipes']))
    baseline_time = report['phases']['1']['wall_seconds']
    candidates = []
    for n in (2, 3):
        name = f'verified-{n}' if serial else str(n)
        phase = report['phases'].get(name)
        if (phase and not phase['failures'] and phase['reserve_respected'] and
                phase['serial_policy'] == serial and not (set(phase['parity_failures']) - set(serial)) and
                baseline_time / phase['wall_seconds'] >= 1.10):
            candidates.append((n, phase['wall_seconds'], name))
    chosen = min(candidates, key=lambda item: item[1]) if candidates else (1, baseline_time, '1')
    selected = chosen[0]
    if len(serial) == len(recipes):
        selected = 1
    settings = {'max_concurrent_jobs': selected, 'gpu_reserve_mib': 2048, 'cpu_threads': 8,
                'memory_mib': report['memory_mib'], 'serial_recipes': serial,
                'measured_policy_phase': chosen[2],
                'calibration_report': str(output / 'calibration.json')}
    report['selected'] = settings
    atomic_json(output / 'calibration.json', report, replace=True)
    return settings
