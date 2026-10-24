"""Sequential train/evaluate DAG; unavailable datasets remain visible."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import statistics

from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.cross import load_dataset, pair_ready, cross_predict
from ntv_ai.experiments.recipes import resolve_recipe, compatibility, validate_recipe
from ntv_ai.experiments.runs import Run, now
from ntv_ai.storage import atomic_json, digest, file_hash, load_json


def build_plan(config_path, catalog_path, storage_root=None):
    config_path, catalog_path = Path(config_path).resolve(), Path(catalog_path).resolve()
    cfg = read_config(config_path)
    if cfg.get('schema_version') != 2 or cfg.get('purpose') != 'exploratory':
        raise ValueError('Matriz v2 requer finalidade exploratory')
    seeds = cfg['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds inválidas')
    if cfg['operation'] not in ('train','smoke') or cfg.get('evaluation_split','test') != 'test':
        raise ValueError('Matriz científica usa teste; pilotos usam smoke sem nós de teste')
    catalog = load_json(catalog_path)
    if catalog['purpose'] != cfg['purpose']:
        raise ValueError('Finalidades divergentes')
    entries = catalog['datasets']
    if storage_root:
        old_root = Path(catalog.get('storage_root', catalog_path.parent.parent.parent)).resolve()
        new_root = Path(storage_root).resolve()
        # Shared data paths are already absolute on the preparation host. Only
        # relocate descriptors when moving the catalog to another checkout.
        for datasets in entries.values() if new_root != old_root else []:
            for entry in datasets.values():
                old_manifest = Path(entry['manifest']['path'])
                entry['manifest']['path'] = str(new_root / old_manifest.relative_to(old_root))
                manifest, _ = load_dataset(entry)
                entry['source_roots'] = {name:str(new_root/Path(source['root']).relative_to(old_root))
                                        for name,source in manifest['sources'].items()}
    experiments, pair_cache = [], {}
    for origin in cfg['origins']:
        for protocol, recipes in cfg['recipes'].items():
            source = entries[origin][protocol]
            manifest, records = load_dataset(source)
            ready = manifest['view']['eligible'] and all(any(r.split == s for r in records) for s in ('train','val'))
            for recipe in recipes:
                resolved = resolve_recipe(config_path.parent / recipe, storage_root)
                resolved['dataset'] = {k:v for k,v in source.items() if k not in ('eligible','blockers','counts')}
                resolved['purpose'] = 'exploratory'
                for seed in seeds:
                    config = deepcopy(resolved)
                    config['training']['seed'] = seed
                    if 'epochs' in cfg: config['training']['epochs'] = cfg['epochs']
                    validate_recipe(config)
                    node_id = f"{origin}/{protocol}/{config['model']['backend']}-{config['model']['variant']}/{seed}"
                    item = {'id':node_id,'kind':'train','origin':origin,'protocol':protocol,
                            'backend':config['model']['backend'],'variant':config['model']['variant'],
                            'seed':seed,'config':config,
                            'state':'pending' if ready else 'blocked'}
                    if not ready: item['reason'] = 'source_ineligible_or_missing_train_val'
                    experiments.append(item)
                    for destination in cfg.get('targets_by_origin', {}).get(origin, cfg.get('targets', [])):
                        target = entries[destination][protocol]
                        key=(origin,destination,protocol)
                        if key not in pair_cache:
                            pair_cache[key] = pair_ready(source, target)
                        decision = deepcopy(pair_cache[key])
                        if decision['state'] == 'pending' and not ready:
                            decision.update(state='blocked', reason='source_ineligible')
                        experiments.append({'id':node_id+'/test/'+destination,'kind':'evaluate',
                            'origin':origin,'target':destination,'protocol':protocol,'backend':item['backend'],
                            'variant':item['variant'],
                            'seed':seed,'depends_on':node_id,'target_dataset':target,**decision})
    if not experiments or len({n['id'] for n in experiments}) != len(experiments):
        raise ValueError('Matriz vazia ou com identidades de execução duplicadas')
    return {'schema_version':2,'purpose':cfg['purpose'],'operation':cfg['operation'],
            **({'execution':cfg['execution']} if 'execution' in cfg else {}),
            'config_sha256':file_hash(config_path),'catalog_sha256':file_hash(catalog_path),
            'experiments':experiments,'dataset_readiness':{k:{p:{'eligible':v['eligible'],
                'blocker_count':len(v['blockers']), 'classes':v['classes']} for p,v in ds.items()} for k,ds in entries.items()}}


def summarize_matrix(path):
    path=Path(path); state=load_json(path/'campaign.json')
    groups={}
    for item in state['experiments']:
        if item['state'] != 'completed' or not item.get('result',{}).get('metrics'):
            continue
        result=item['result'];metrics=result['metrics']
        if metrics['mAP50_95'] is None: continue
        env=load_json(Path(result['run_path'])/'runtime.json')
        environment={k:env.get(k) for k in ('dependencies','gpu','torch','cuda','python','code_digest')}
        identity={'source_recipe':result['compatibility'],'target':result.get('target_version',metrics['dataset_version']),
                  'classes':result.get('classes',list(metrics['per_class'])),'split':metrics['protocol']['split'],
                  'environment':environment,'origin':item['origin'],'protocol':item['protocol']}
        key=digest(identity)
        group=groups.setdefault(key,{'identity':identity,'backend':item['backend'],'origin':item['origin'],
            'target':item.get('target',item['origin']),'protocol':item['protocol'],'seeds':[],'metrics':[]})
        if item['seed'] in group['seeds']:raise ValueError('Seed duplicada na agregação')
        group['seeds'].append(item['seed']);group['metrics'].append(metrics)
    def aggregate(values):
        values=[v for v in values if v is not None]
        return {'n':len(values),'mean':statistics.mean(values) if values else None,
                'sample_std':statistics.stdev(values) if len(values)>1 else None}
    for group in groups.values():
        rows=group.pop('metrics')
        group['aggregate']={k:aggregate([r[k] for r in rows]) for k in ('mAP50_95','AP50','AP75','AR50_95')}
        group['per_class']={c:{k:aggregate([r['per_class'][c][k] for r in rows]) for k in rows[0]['per_class'][c]}
                            for c in rows[0]['per_class']}
        group['AP_by_size']={k:aggregate([r['AP_by_size'][k] for r in rows]) for k in rows[0]['AP_by_size']}
    counts=dict(Counter(x['state'] for x in state['experiments']))
    report={'purpose':state['purpose'],'counts':counts,'groups':groups,
            'complete':not any(counts.get(k) for k in ('pending','running','interrupted','blocked','failed')),
            'blocked':[{'id':x['id'],'reason':x.get('reason')} for x in state['experiments'] if x['state']=='blocked'],
            'failures':[{'id':x['id'],'error':x.get('error')} for x in state['experiments'] if x['state'] in ('failed','interrupted')],
            'dataset_readiness':state['dataset_readiness']}
    atomic_json(path/'summary.json',report,replace=True)
    return report


def matrix_campaign(config_path, catalog=None, storage_root=None, runs_root=None, resume=None, plan_only=False):
    from ntv_ai.pipelines.experiments import launch
    settings = load_json(Path(resume)/'campaign.json') if resume else read_config(config_path)
    if settings.get('execution') and not plan_only:
        from ntv_ai.experiments.scheduler import concurrent_campaign
        return concurrent_campaign(config_path, catalog, storage_root, runs_root, resume)
    if resume:
        path=Path(resume).resolve();state=load_json(path/'campaign.json');context=None
        if any(x['state']=='running' for x in state['experiments']):
            raise ValueError('Worker running precisa ser reconciliado antes de retomar')
    else:
        if catalog is None:raise ValueError('Informe --catalog da preparação imutável')
        state=build_plan(config_path,catalog,storage_root)
        source=next(x for x in state['experiments'] if x['kind']=='train')['config']
        context=Run(Path(runs_root or Path(source['runtime']['storage_root'])/'runs'),
                    'campaign_plan' if plan_only else 'matrix_campaign',read_config(config_path))
        path=context.path;atomic_json(path/'campaign.json',state)
    index={x['id']:x for x in state['experiments']}
    try:
        if not plan_only:
            for item in state['experiments']:
                if item['state'] == 'blocked' and item.get('reason') == 'training_dependency_not_completed':
                    if index[item['depends_on']]['state'] == 'completed':
                        item.update(state='pending')
                        item.pop('reason', None)
                if item['state'] not in ('pending','interrupted'):continue
                if item['kind']=='evaluate' and index[item['depends_on']]['state']!='completed':
                    item.update(state='blocked',reason='training_dependency_not_completed')
                    atomic_json(path/'campaign.json',state,replace=True);continue
                continuation = None
                if item['kind'] == 'train' and item['state'] == 'interrupted' and item.get('attempts'):
                    parent = Path(item['attempts'][-1])
                    if (parent/'progress.json').exists():
                        continuation = parent
                item.update(state='running',started_at=now());atomic_json(path/'campaign.json',state,replace=True)
                try:
                    if item['kind']=='train':
                        def linked(created):
                            item.setdefault('attempts', []).append(str(created))
                            atomic_json(path/'campaign.json',state,replace=True)
                        result=launch(state['operation'],item['config'],runs_root=runs_root,
                                      resume=continuation, on_created=linked)
                    else:
                        parent=index[item['depends_on']]['result']['run_path']
                        result=cross_predict(parent,item['target_dataset'],runs_root=runs_root)
                    item.update(result=result,state=result.get('state','completed'))
                except Exception as exc:
                    item.update(state='failed',error=str(exc))
                except KeyboardInterrupt:
                    item.update(state='interrupted',error='Interrompido pelo usuário');raise
                finally:
                    item['finished_at']=now();atomic_json(path/'campaign.json',state,replace=True)
        report=summarize_matrix(path)
        if context:context.__exit__(None,None,None)
        return {'campaign':str(path),'counts':report['counts'],'complete':report['complete']}
    except BaseException as exc:
        if context:context.__exit__(type(exc),exc,exc.__traceback__)
        raise
