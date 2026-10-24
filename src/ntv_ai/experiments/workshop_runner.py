"""Serial, deadline-aware driver. A completed node always has a verifiable result."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tarfile
import time

from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.cross import load_dataset
from ntv_ai.experiments.recipes import resolve_recipe
from ntv_ai.experiments.workshop_prompts import bank
from ntv_ai.pipelines.benchmark_data import descriptor
from ntv_ai.pipelines.workshop_data import prepare_workshop
from ntv_ai.storage import atomic_json, digest, file_hash, load_json, write_snapshot


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def process_alive(identity):
    if not identity:
        return False
    try:
        text = Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
        return text[0] != 'Z' and text[19] == identity['start_ticks'] and (
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot_id'])
    except FileNotFoundError:
        return False


def training_smoke(data, folder):
    path = folder / 'manifest.json'
    if path.exists():
        return descriptor(path)
    manifest, rows = load_dataset(data)
    original_ids = set(manifest['view']['evaluation_image_ids'])
    selected = sorted((r for r in rows if r.image_id in original_ids and r.split == 'train'), key=lambda r:r.image_id)[:4]
    selected = deepcopy(selected)
    for index, row in enumerate(selected):
        row.split = 'train' if index < 2 else 'val'
        row.group_id = 'software-smoke-' + row.image_id
        row.metadata['grouping_basis'] = 'Software-only repartition of training data; not independent evaluation'
    view = deepcopy(manifest['view'])
    view.update(id=view['id'] + '-software-smoke', image_ids=[r.image_id for r in selected],
                evaluation_image_ids=[r.image_id for r in selected], purpose='exploratory')
    return descriptor(write_snapshot(folder, selected, manifest['sources'],
        {'operation': 'software_smoke_train_images_only', 'parent': manifest['version']}, [], view=view))


class Driver:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.root = self.config_path.parents[2]
        self.cfg = read_config(self.config_path)
        self.workspace = Path(self.cfg['workspace'])
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.path = self.workspace / 'orchestration.json'
        self.state = load_json(self.path) if self.path.exists() else {
            'config_sha256': file_hash(self.config_path), 'phase': 'preparation', 'nodes': {},
            'vlm_seconds': 0., 'complete': False, 'created_at': datetime.now().astimezone().isoformat()}
        if self.state['config_sha256'] != file_hash(self.config_path):
            raise ValueError('Campaign configuration changed; create another workspace')
        self.state['expected'] = {'training': 18, 'training_evaluations': 54, 'vlm_validation': 16,
                                  'vlm_test': 'P0 and selected prompt for each model/protocol; deduplicated'}
        self.hard_deadline = timestamp(self.cfg['deadline'])
        self.compute_deadline = self.hard_deadline - self.cfg['report_reserve_seconds']
        self.assets = None

    def save(self):
        self.state['updated_at'] = datetime.now().astimezone().isoformat()
        atomic_json(self.path, self.state, replace=True)

    def node(self, key, request, deadline, vlm=False):
        folder = self.workspace / 'jobs' / key
        request = {**request, 'output': str(folder)}
        identity = digest(request)
        item = self.state['nodes'].setdefault(key, {'state': 'pending', 'identity': identity,
                                                   'request': request, 'attempts': []})
        if item['identity'] != identity:
            raise ValueError(f'Job changed: {key}')
        result_path = folder / 'result.json'
        if item['state'] == 'completed':
            if not result_path.exists() or file_hash(result_path) != item['result_sha256']:
                raise ValueError(f'Completed result changed: {key}')
            return load_json(result_path)
        if item['state'] in ('failed', 'blocked'):
            return None
        if process_alive(item.get('process')):
            raise RuntimeError(f'Existing worker still active: {key}; wait for it instead of duplicating')
        if result_path.exists():
            item.update(state='completed', result_sha256=file_hash(result_path), recovered=True)
            self.save()
            return load_json(result_path)
        if time.time() >= deadline:
            item.update(state='budget_exhausted', reason='time_limit')
            self.save()
            return None
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'request.json', request, replace=True)
        python = self.assets['python'] if vlm else sys.executable
        module = 'ntv_ai.pipelines.workshop_vlm_worker' if vlm else 'ntv_ai.pipelines.workshop_job'
        env = dict(os.environ, PYTHONPATH=str(self.root/'src'), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', PYTHONHASHSEED='42', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        log = folder / f'worker-{len(item["attempts"])+1}.log'
        started = time.time()
        with log.open('w') as stream:
            child = subprocess.Popen([python, '-u', '-m', module, str(folder/'request.json')],
                cwd=self.root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            stat = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
            item.update(state='running', process={'pid': child.pid, 'start_ticks': stat[19],
                        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()},
                        started_at=started)
            item['attempts'].append(str(log))
            self.state['current'] = key
            self.save()
            print(f'START {key}', flush=True)
            exhausted = False
            try:
                while child.poll() is None:
                    if time.time() >= deadline:
                        exhausted = True
                        os.killpg(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        break
                    time.sleep(2)
            except BaseException:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                item['state'] = 'interrupted'
                self.save()
                raise
        elapsed = time.time() - started
        if vlm:
            self.state['vlm_seconds'] += elapsed
        item.update(elapsed_seconds=item.get('elapsed_seconds',0.) + elapsed, finished_at=time.time())
        if result_path.exists() and child.returncode == 0:
            item.update(state='completed', result_sha256=file_hash(result_path))
        else:
            item.update(state='budget_exhausted' if exhausted else 'failed', returncode=child.returncode,
                        reason='time_limit' if exhausted else 'worker_failure', log=str(log))
        self.save()
        self.report()
        print(f'END {key}: {item["state"]} ({elapsed:.1f}s)', flush=True)
        return load_json(result_path) if item['state'] == 'completed' else None

    def train_config(self, arm, protocol, seed):
        cfg = resolve_recipe(self.root/f'configs/benchmark/rfdetr-nano-{protocol}.yaml', self.root)
        cfg['dataset'] = self.prepared['arms'][arm][protocol]
        cfg['training']['seed'] = seed
        cfg['runtime'].update(cpu_threads=8)
        return cfg

    def vf(self, model, protocol, prompt, split, **extra):
        return {'model': model, 'protocol': protocol, 'prompt_id': prompt, 'split': split,
                'dataset': self.prepared['targets']['ntv_ai'][protocol],
                'asset': self.assets['models'][model], 'resolution': 640 if protocol == 'full640' else 1280, **extra}

    def run_seed(self, seed):
        self.state['phase'] = f'training_seed_{seed}'
        for protocol in self.cfg['protocols']:
            for arm in self.cfg['arms']:
                key = f'train/{seed}/{protocol}/{arm}'
                result = self.node(key, {'kind': 'train', 'config': self.train_config(arm, protocol, seed)}, self.compute_deadline)
                if result:
                    self.evaluate_training(key, protocol)

    def evaluate_training(self, source, protocol):
        for target in self.cfg['targets']:
            self.node(source.replace('train/', 'evaluate/', 1) + '/' + target,
                {'kind': 'evaluate', 'source_result': str(self.workspace/'jobs'/source/'result.json'),
                 'target': self.prepared['targets'][target][protocol]}, self.hard_deadline)

    def vlms(self):
        self.state['phase'] = 'vlm'
        for model in self.cfg['models']:
            for protocol in self.cfg['protocols']:
                smoke_key = f'smoke/{model}/{protocol}'
                if self.state['nodes'].get(smoke_key,{}).get('state') != 'completed':
                    continue
                rows = []
                for prompt in range(4):
                    deadline = min(self.compute_deadline, time.time()+max(0., self.cfg['vlm_budget_seconds']-self.state['vlm_seconds']))
                    result = self.node(f'vlm/{model}/{protocol}/val/{prompt}',
                        self.vf(model, protocol, prompt, 'val'), deadline, vlm=True)
                    if result:
                        rows.append(result)
                if len(rows) != 4:
                    self.state.setdefault('vlm_blockers', {})[f'{model}/{protocol}'] = 'all_four_validation_prompts_required'
                    self.save()
                    continue
                self.state.setdefault('vlm_blockers', {}).pop(f'{model}/{protocol}', None)
                best = max(rows, key=lambda r:(r['metrics']['mAP50_95'], -r['prompt_id']))['prompt_id']
                selection = {'selected_prompt': best, 'metric': 'mAP50_95', 'selection_split': 'val',
                             'validation_results': {str(r['prompt_id']): r['identity'] for r in rows}}
                selection_path = self.workspace/'selections'/model/f'{protocol}.json'
                if selection_path.exists() and load_json(selection_path) != selection:
                    raise ValueError('Frozen prompt selection changed')
                if not selection_path.exists():
                    atomic_json(selection_path, selection)
                for prompt in dict.fromkeys([0, best]):
                    validation = next(r for r in rows if r['prompt_id'] == prompt)
                    deadline = min(self.compute_deadline, time.time()+max(0., self.cfg['vlm_budget_seconds']-self.state['vlm_seconds']))
                    self.node(f'vlm/{model}/{protocol}/test/{prompt}',
                        self.vf(model, protocol, prompt, 'test', threshold=validation['operating']['threshold']),
                        deadline, vlm=True)

    def report(self):
        groups, vlms = defaultdict(list), []
        for key, item in self.state['nodes'].items():
            if item['state'] != 'completed':
                continue
            result = load_json(Path(item['request']['output'])/'result.json')
            if key.startswith('evaluate/') and 'metrics' in result:
                _, seed, protocol, arm, target = key.split('/')
                groups[(protocol,arm,target)].append({'seed': int(seed), 'metrics': result['metrics']})
            if key.startswith('vlm/') and item['request']['split'] == 'test':
                vlms.append(result)
        aggregates = []
        for (protocol,arm,target), rows in groups.items():
            def summarize(values):
                values = [v for v in values if v is not None]
                return {'n':len(values), 'mean':statistics.mean(values) if values else None,
                        'sample_std':statistics.stdev(values) if len(values)>1 else None}
            aggregates.append({'protocol':protocol, 'arm':arm, 'target':target,
                'seeds':[r['seed'] for r in rows],
                'metrics':{k:summarize([r['metrics'][k] for r in rows]) for k in ('mAP50_95','AP50','AR50_95')},
                'per_class':{c:{k:summarize([r['metrics']['per_class'][c][k] for r in rows])
                    for k in ('mAP50_95','AP50','AR50_95')} for c in self.cfg['classes']}})
        result = {'counts':dict(Counter(x['state'] for x in self.state['nodes'].values())),
                  'complete':self.state['complete'], 'training_aggregates':aggregates, 'vlm_tests':vlms,
                  'nodes':self.state['nodes'], 'deadline':self.cfg['deadline'],
                  'limitations':['8 NTV test images from a single capture', 'All mixture arms use NTV validation',
                      'Different training image counts and optimizer steps', 'Qwen scores are uncalibrated generation heuristics',
                      'GPL-3.0 repository notices for Furnas/PLAD; no license equivalence claim']}
        atomic_json(self.workspace/'results.json',result,replace=True)
        lines=['# Campanha complementar NTV-AI', '', f"Prazo: {self.cfg['deadline']}",
               f"Concluída: {self.state['complete']}", f"Estados: {result['counts']}", '',
               'Todos os braços da mistura usam validação NTV-AI. Não são transferência sem contato com o alvo.', '',
               '| Protocolo | Treino | Teste | Seeds | mAP50–95 | Desvio |', '|---|---|---|---|---|---|']
        for r in aggregates:
            m=r['metrics']['mAP50_95']
            lines.append(f"| {r['protocol']} | {r['arm']} | {r['target']} | {r['seeds']} | {m['mean']} | {m['sample_std']} |")
        lines += ['', '## VLMs', '', '| Modelo | Protocolo | Prompt | mAP50–95 (7 classes) | mAP (3 classes) | Falhas de saída |', '|---|---|---|---|---|---|']
        for r in vlms:
            lines.append(f"| {r['model']} | {r['protocol']} | {r['prompt_id']} | {r['metrics']['mAP50_95']} | {r['subset_metrics']['mAP50_95']} | {len(r['invalid_samples'])} |")
        lines += ['', '## Pendências', '']
        lines += [f"- {key}: {v['state']} ({v.get('reason','')})" for key,v in self.state['nodes'].items() if v['state']!='completed']
        lines += ['', '## Limitações', '', *['- '+x for x in result['limitations']]]
        (self.workspace/'relatorio.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        with (self.workspace/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.prepared = prepare_workshop(self.cfg,self.workspace)
            atomic_json(self.workspace/'preparation.json',self.prepared,replace=True)
            frozen_prompts = self.workspace/'prompts.json'
            if frozen_prompts.exists() and load_json(frozen_prompts)!=bank():
                raise ValueError('Frozen prompts changed')
            if not frozen_prompts.exists():
                atomic_json(frozen_prompts,bank())
            self.state['phase']='preflight'
            self.save()
            cfg=self.train_config('ntv_only','full640',42)
            cfg['dataset']=training_smoke(cfg['dataset'],self.workspace/'software-smoke-data')
            cfg['training']['epochs']=1
            smoke=self.node('smoke/rfdetr',{'kind':'train','config':cfg},min(self.compute_deadline,time.time()+1800))
            if not smoke:
                raise RuntimeError('RF-DETR preflight did not pass')
            while not (self.workspace/'assets-ready.json').exists():
                if time.time() >= self.compute_deadline:
                    raise RuntimeError('Model acquisition exceeded campaign deadline')
                probe = subprocess.run(['tmux','has-session','-t','ntv-workshop-assets'],capture_output=True)
                if probe.returncode:
                    raise RuntimeError('Model acquisition is incomplete and its tmux session has exited; inspect acquisition.log')
                self.state['phase']='waiting_for_model_acquisition'
                self.save()
                time.sleep(10)
            ready = load_json(self.workspace/'assets-ready.json')
            if ready['assets_sha256'] != file_hash(self.workspace/'assets.json'):
                raise ValueError('Model assets manifest changed')
            self.assets = load_json(self.workspace/'assets.json')
            self.state['phase']='vlm_preflight'
            for model in self.cfg['models']:
                for protocol in self.cfg['protocols']:
                    self.node(f'smoke/{model}/{protocol}',self.vf(model,protocol,3,'train',smoke=True),
                              min(self.compute_deadline,time.time()+1800),vlm=True)
            self.freeze_implementation()
            self.run_seed(42)
            self.vlms()
            for seed in (43,44):
                self.run_seed(seed)
            # Retry pending evaluations at the end, without reopening any completed result.
            for key,item in list(self.state['nodes'].items()):
                if key.startswith('train/') and item['state']=='completed':
                    self.evaluate_training(key,key.split('/')[2])
            required_trains=sum(k.startswith('train/') and v['state']=='completed' for k,v in self.state['nodes'].items())
            required_evals=sum(k.startswith('evaluate/') and v['state']=='completed' for k,v in self.state['nodes'].items())
            validations=sum(k.startswith('vlm/') and '/val/' in k and v['state']=='completed' for k,v in self.state['nodes'].items())
            self.state['complete']=(required_trains==18 and required_evals==54 and validations==16 and
                all(v['state']=='completed' for v in self.state['nodes'].values()) and not self.state.get('vlm_blockers'))
            self.state['phase']='complete' if self.state['complete'] else 'incomplete'
            self.state.pop('current',None)
            self.save()
            self.report()
            if self.state['complete']:
                (self.workspace/'COMPLETE').touch()

    def freeze_implementation(self):
        files = sorted({*self.root.glob('src/ntv_ai/**/*.py'), *self.root.glob('configs/benchmark/*.yaml'),
                        *self.root.glob('configs/experiments/**/*.yaml'), *self.root.glob('scripts/*workshop*.py'),
                        self.root/'scripts/start_workshop_campaign.sh'})
        snapshot = {str(p.relative_to(self.root)):file_hash(p) for p in files}
        identity = {'code':snapshot,'config_sha256':file_hash(self.config_path),
                    'prompts_sha256':file_hash(self.workspace/'prompts.json'),
                    'assets_sha256':file_hash(self.workspace/'assets.json'),
                    'preparation_identity':self.prepared['identity']}
        path = self.workspace/'frozen-protocol.json'
        if path.exists():
            if load_json(path) != identity:
                raise ValueError('Frozen scientific implementation changed; inspect before resuming')
            return
        with tarfile.open(self.workspace/'implementation.tar.gz','w:gz') as archive:
            for p in files:
                archive.add(p,arcname=str(p.relative_to(self.root)))
        atomic_json(path,identity)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/benchmark/workshop-campaign.yaml'))
    driver=Driver(parser.parse_args().config)
    try:
        driver.run()
    except BlockingIOError:
        # Another supervisor owns the ledger. Never overwrite its live state.
        raise SystemExit('Workshop supervisor already running')
    except BaseException as exc:
        driver.state.update(complete=False,last_error=f'{type(exc).__name__}: {exc}')
        driver.save()
        driver.report()
        raise


if __name__=='__main__':
    main()
