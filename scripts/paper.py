"""Portable entry points for the published experiments; no implicit GPU work."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
import math
import statistics
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ntv_ai.storage import atomic_json, digest, file_hash, load_json, read_snapshot, write_snapshot
from ntv_ai.domain.records import ImageRecord

PAPER=ROOT/'paper'
METRICS=['mAP50_95','AP50','AP75','AR50_95']


def catalog():return load_json(PAPER/'experiments.json')
def experiment(identifier):
    matches=[x for x in catalog()['training'] if x['id']==identifier]
    if len(matches)!=1:raise ValueError(f'Unknown experiment: {identifier}; use list')
    return matches[0]


def frozen(key):
    folder=PAPER/'datasets'/key
    manifest=load_json(folder/'manifest.json')
    raw=gzip.decompress((folder/'images.jsonl.gz').read_bytes())
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=manifest['records_sha256']:
        raise ValueError(f'Frozen records changed: {key}')
    # Reuse the project's complete snapshot integrity checks on temporary files.
    with tempfile.TemporaryDirectory(prefix='ntv-snapshot-') as tmp:
        p=Path(tmp)
        (p/'images.jsonl').write_bytes(raw)
        (p/'manifest.json').write_bytes((folder/'manifest.json').read_bytes())
        (p/'issues.json').write_bytes((folder/'issues.json').read_bytes())
        return read_snapshot(p/'manifest.json')


def close_metrics(actual,expected,where):
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k in ('dataset_version','predictions_sha256','evaluated_at'):continue
            if k not in actual:raise AssertionError(f'{where}: missing {k}')
            close_metrics(actual[k],v,where+'/'+k)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-10):
            raise AssertionError(f'{where}: {actual} != {expected}')
    elif actual!=expected:raise AssertionError(f'{where}: {actual!r} != {expected!r}')


def replay(output=None):
    """Recompute the main paper's AP from saved predictions on CPU, without images."""
    from ntv_ai.evaluation.bbox import evaluate
    entries=load_json(PAPER/'datasets/index.json')
    key=next(k for k,v in entries.items() if v['view_id']=='expanded18')
    manifest,records=frozen(key)
    assert len(records)==18 and sum(len(r.annotations) for r in records)==411
    groups=defaultdict(list)
    for item in catalog()['training']:
        if item['study']!='main':continue
        folder=PAPER/'results/main'/item['id'].split('/',1)[1]
        expected=load_json(folder/'result.json')['expanded18']
        with tempfile.TemporaryDirectory(prefix='ntv-metric-replay-') as tmp:
            p=Path(tmp)/'predictions.jsonl'
            p.write_bytes(gzip.decompress((folder/'predictions-expanded18.jsonl.gz').read_bytes()))
            actual=evaluate(manifest,records,p,{'split':'test','max_detections':100})
        close_metrics(actual,expected,item['id'])
        groups[(item['model'],item['protocol'])].append(actual)
    rows=[]
    for (model,protocol),results in sorted(groups.items()):
        assert len(results)==3
        rows.append({'model':model,'protocol':protocol,'seeds':[42,43,44],
            'metrics':{metric:{'mean':statistics.mean(r[metric]*100 for r in results),
                               'sample_std':statistics.stdev(r[metric]*100 for r in results)}
                       for metric in METRICS},
            'per_class':{c:{metric:{'mean':statistics.mean(r['per_class'][c][metric]*100 for r in results),
                                   'sample_std':statistics.stdev(r['per_class'][c][metric]*100 for r in results)}
                           for metric in ('mAP50_95','AP50','AR50_95')}
                         for c in results[0]['per_class']}})
    result={'images':18,'objects':411,'replayed_runs':18,'scale':'0-100','results':rows}
    if output:
        atomic_json(Path(output),result)
    return result


def verify():
    checksums=PAPER/'SHA256SUMS.json'
    if checksums.exists():
        for name,expected in load_json(checksums).items():
            if file_hash(ROOT/name)!=expected:raise ValueError(f'Release artifact changed: {name}')
    ids=load_json(PAPER/'datasets/index.json')
    for key in ids:frozen(key)
    counts=Counter(x['study'] for x in catalog()['training'])
    assert counts=={'main':18,'external':72,'mixed':18,'annotation':1}
    assert len(catalog()['vlm'])==21
    result=replay()
    annotation=replay_annotation()
    print(json.dumps({'verified':True,'snapshots':len(ids),'training_configs':dict(counts),
                      'vlm_configs':21,'metric_replays':result['replayed_runs'],
                      'annotation_replays':len(annotation)},indent=2))


def replay_annotation():
    """Replay the five saved development comparisons, including supplementary S9."""
    from ntv_ai.evaluation.bbox import evaluate
    item=next(x for x in catalog()['training'] if x['study']=='annotation')
    manifest,records=frozen(item['config']['dataset_key'])
    expected=load_json(PAPER/'results/annotation-combination.json')
    names={'combined-export-threshold':'combined_export_threshold',
           'current-export-threshold':'candidate_export_threshold',
           'combined-reference':'combined_reference',
           'combined-operational':'combined_operational','full640':'full640'}
    output={}
    for name,key in names.items():
        with tempfile.TemporaryDirectory(prefix='ntv-annotation-replay-') as tmp:
            path=Path(tmp)/'predictions.jsonl'
            path.write_bytes(gzip.decompress((PAPER/'results/annotation-predictions'/f'{name}.jsonl.gz').read_bytes()))
            actual=evaluate(manifest,records,path,{'split':'val','max_detections':100})
        close_metrics(actual,expected[key],'annotation/'+key)
        output[key]=actual
    return output


def required(study):
    pairs=set()
    for item in catalog()['training']:
        if study not in ('all',item['study']):continue
        pairs.add((item['config']['dataset_key'],item['protocol']))
        pairs.update((t['dataset_key'],t['protocol']) for t in item['targets'])
    if study in ('all','vlm'):
        pairs.update((x['dataset_key'],x['protocol']) for x in catalog()['vlm'])
    return sorted(pairs)


def prepare_one(key,protocol,roots,output,workers=1):
    from ntv_ai.pipelines.prepare import materialize_tiles
    from ntv_ai.pipelines.benchmark_data import descriptor
    manifest,records=frozen(key)
    originals=[r for r in records if not r.metadata.get('transform')]
    active_ids=set(manifest['view']['evaluation_image_ids'])
    active=[r for r in originals if r.image_id in active_ids]
    sources={}
    for source in {r.dataset_id for r in originals}:
        if source not in roots:raise ValueError(f'Missing image root: {source}')
        path=Path(roots[source]).expanduser().resolve()
        sources[source]=deepcopy(manifest['sources'][source]);sources[source]['root']=str(path)
    for r in active:
        path=(Path(sources[r.dataset_id]['root'])/r.path).resolve()
        if not path.is_relative_to(Path(sources[r.dataset_id]['root'])):
            raise ValueError('Image path leaves its source root')
        if not path.is_file() or file_hash(path)!=r.sha256:
            raise ValueError(f'Missing or different image: {path}')
    folder=output/(key+'-'+protocol)
    identity=digest({'key':key,'protocol':protocol,'roots':roots,'source':manifest['version']})
    if (folder/'identity.json').exists():
        if load_json(folder/'identity.json')['identity']!=identity:
            raise ValueError('Preparation roots changed; use a new output directory')
        return descriptor(folder/'snapshot/manifest.json')
    if folder.exists():raise ValueError(f'Incomplete preparation; use a new output directory: {folder}')
    folder.mkdir(parents=True)
    view=deepcopy(manifest['view'])
    view['tiling']={'enabled':protocol=='tiles1280','size':1280,'overlap':0.25,'border_policy':'clip'}
    samples=active
    if protocol=='tiles1280':
        samples=materialize_tiles(active,sources,view['classes'],view['tiling'],folder/'tiles',workers=workers)
        expected={r.image_id:r for r in records if r.metadata.get('transform')}
        if expected:
            assert {r.image_id for r in samples}==set(expected),'Tile identity mismatch'
            for r in samples:
                if r.sha256!=expected[r.image_id].sha256:
                    raise ValueError('Regenerated tile bytes differ; use the pinned Pillow version')
                assert [asdict(a.geometry) for a in r.annotations]==[asdict(a.geometry) for a in expected[r.image_id].annotations]
    view.update(id=view['id']+'-portable-'+protocol,image_ids=[r.image_id for r in samples])
    stored=originals+samples if protocol=='tiles1280' else originals
    path=write_snapshot(folder/'snapshot',stored,sources,
        {'operation':'paper_relocation','public_frozen_version':manifest['version']},[],view=view)
    atomic_json(folder/'identity.json',{'identity':identity})
    return descriptor(path)


def prepare(args):
    roots=load_json(args.roots)
    roots={k:str((args.roots.resolve().parent/Path(v).expanduser()).resolve()) for k,v in roots.items()}
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    path=output/'catalog.json'
    result=load_json(path) if path.exists() else {'datasets':{}}
    for key,protocol in required(args.study):
        print('Preparing',key,protocol,flush=True)
        result['datasets'][key+':'+protocol]=prepare_one(key,protocol,roots,output,args.workers)
        atomic_json(path,result,replace=True)
    print(path)


def prepared_entry(path,key,protocol):
    entry=load_json(path)['datasets'][key+':'+protocol]
    # Validate the descriptor and snapshot before launching workers.
    if file_hash(Path(entry['manifest']['path']))!=entry['manifest']['sha256']:
        raise ValueError('Prepared manifest changed')
    read_snapshot(Path(entry['manifest']['path']))
    return entry


def export_descriptor(args):
    entries=load_json(PAPER/'datasets/index.json')
    keys=[k for k,v in entries.items() if v['view_id']==args.view]
    if len(keys)!=1:raise ValueError('Select an exact, unique frozen view identifier')
    entry=prepared_entry(args.prepared,keys[0],args.protocol)
    atomic_json(args.output,entry)
    print(args.output)


def resolved(item,args):
    cfg=deepcopy(item['config'])
    key=cfg.pop('dataset_key')
    cfg['dataset']=prepared_entry(args.prepared,key,item['protocol'])
    cfg['runtime']['storage_root']=str(ROOT)
    if getattr(args,'python',None):cfg['runtime']['python']=str(args.python.resolve())
    return cfg


def train(args):
    item=experiment(args.id);cfg=resolved(item,args)
    if args.dry_run:
        print(json.dumps(cfg,indent=2));return
    from ntv_ai.pipelines.experiments import launch
    result=launch('train',cfg,runs_root=args.runs,resume=args.resume)
    print(json.dumps(result,indent=2))


def evaluate_run(args):
    item=experiment(args.id)
    from ntv_ai.experiments.cross import cross_predict
    for target in item['targets']:
        descriptor=prepared_entry(args.prepared,target['dataset_key'],target['protocol'])
        result=cross_predict(args.source_run,descriptor,runs_root=args.runs,split=target['split'])
        if result.get('state','completed')!='completed':raise RuntimeError(result)
        print(json.dumps(result,indent=2))


def vlm(args):
    item=next(x for x in catalog()['vlm'] if x['id']==args.id)
    request=deepcopy(item);request.pop('id')
    request['dataset']=prepared_entry(args.prepared,request.pop('dataset_key'),request['protocol'])
    request['output']=str(args.output.resolve())
    request['asset']['path']=str(args.asset.resolve())
    for name,expected in request['asset']['files'].items():
        if file_hash(args.asset/name)!=expected:raise ValueError('Model asset mismatch: '+name)
    if args.dry_run:print(json.dumps(request,indent=2));return
    from ntv_ai.pipelines.workshop_vlm_worker import work
    print(json.dumps(work(request),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list');sub.add_parser('verify');sub.add_parser('replay-annotation')
    p=sub.add_parser('replay');p.add_argument('--output',type=Path)
    p=sub.add_parser('descriptor');p.add_argument('--view',required=True)
    p.add_argument('--protocol',choices=['full640','tiles1280'],required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare');p.add_argument('--roots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--study',choices=['all','main','external','mixed','annotation','vlm'],default='main')
    for name in ['train','evaluate']:
        p=sub.add_parser(name);p.add_argument('--id',required=True)
        p.add_argument('--prepared',type=Path,required=True)
        p.add_argument('--runs',type=Path,default=ROOT/'runs/paper')
        if name=='train':
            p.add_argument('--resume',type=Path);p.add_argument('--python',type=Path)
            p.add_argument('--dry-run',action='store_true')
        else:p.add_argument('--source-run',type=Path,required=True)
    p=sub.add_parser('vlm');p.add_argument('--id',required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.command=='list':
        for x in catalog()['training']:print(x['id'])
        for x in catalog()['vlm']:print(x['id'])
    elif args.command=='verify':verify()
    elif args.command=='replay-annotation':print(json.dumps(replay_annotation(),indent=2))
    elif args.command=='replay':print(json.dumps(replay(args.output),indent=2))
    elif args.command=='descriptor':export_descriptor(args)
    elif args.command=='prepare':prepare(args)
    elif args.command=='train':train(args)
    elif args.command=='evaluate':evaluate_run(args)
    elif args.command=='vlm':vlm(args)


if __name__=='__main__':main()
