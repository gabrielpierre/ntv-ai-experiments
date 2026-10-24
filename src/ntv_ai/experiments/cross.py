"""Read-only pair validation and a separate worker for cross-dataset inference."""
from copy import deepcopy
from pathlib import Path

from ntv_ai.data.audit import audit_records
from ntv_ai.data.views import eligibility
from ntv_ai.domain.records import ArtifactReference
from ntv_ai.models.common import checked_artifact
from ntv_ai.storage import load_json, read_snapshot


def load_dataset(data):
    manifest, records = read_snapshot(checked_artifact(ArtifactReference(**data['manifest'])))
    if manifest['version'] != data['version'] or manifest['view']['id'] != data['view_id'] or manifest['view']['classes'] != data['classes']:
        raise ValueError('Descriptor e manifesto divergentes')
    for key, root in data.get('source_roots', {}).items():
        if key not in manifest['sources']:
            raise ValueError('Locator de fonte desconhecida')
        manifest['sources'][key]['root'] = root
    return manifest, records


def evaluation_data(data, classes, split='test'):
    manifest, records = load_dataset(data)
    view = manifest['view']
    if split not in ('val', 'test') or not classes or not set(classes) <= set(view['classes']):
        raise ValueError('Classes/split de avaliação incompatíveis')
    originals = [r for r in records if r.image_id in set(view['evaluation_image_ids']) and r.split == split]
    samples = [r for r in records if r.image_id in set(view['image_ids']) and r.split == split]
    failures = eligibility(originals, classes) + eligibility(samples, classes)
    if failures:
        raise ValueError(f'Destino inelegível: {failures[:3]}')
    # A corrupt source reference is not cured by selecting fewer classes.
    source_issues = load_json(Path(data['manifest']['path']).parent / manifest['issues'])
    if any('unresolved_source_issue' in i.get('codes', []) for i in source_issues):
        raise ValueError('Destino contém referências de origem não resolvidas')
    # Evaluation does not require the destination's training/validation images.
    selected = {r.image_id:r for r in originals + samples}
    integrity = audit_records(manifest, list(selected.values()))
    if any(i['code'].endswith('integrity_mismatch') for i in integrity['issues']):
        raise ValueError('Integridade do destino divergente')
    evaluated = deepcopy(manifest)
    evaluated['view'].update(classes=list(classes), eligible=True,
        evaluation_image_ids=[r.image_id for r in originals])
    return evaluated, originals, samples


def check_leakage(source_records, target_records):
    def identities(row):
        return {(kind,value) for kind,value in [('group',row.group_id), ('bytes',row.sha256),
                                                ('pixels',row.pixel_sha256)] if value}
    training = set()
    for row in source_records:
        if row.split in ('train','val') and not row.metadata.get('excluded') and not row.metadata.get('transform'):
            training.update(identities(row))
    for row in target_records:
        if training & identities(row):
            raise ValueError(f'Vazamento entre origem e destino: {row.image_id}')


def pair_ready(source, target, split='test'):
    classes = [c for c in source['classes'] if c in target['classes']]
    if not classes:
        return {'state':'not_applicable','classes':[],'reason':'empty_class_intersection'}
    try:
        source_manifest, source_records = load_dataset(source)
        target_manifest, target_records = load_dataset(target)
        selected_ids = set(target_manifest['view']['evaluation_image_ids'])
        selected = [r for r in target_records if r.image_id in selected_ids and r.split == split]
        compatible = [c for c in classes if selected and all(r.annotation_scope.get(c) == 'exhaustive' for r in selected)]
        if not compatible:
            raise ValueError('Destino sem teste com cobertura revisada nas classes compartilhadas')
        excluded = [c for c in classes if c not in compatible]
        classes = compatible
        a = source_manifest['view'].get('tiling', {'enabled':False})
        b = target_manifest['view'].get('tiling', {'enabled':False})
        keys = ('enabled', 'size', 'overlap', 'border_policy') if a.get('enabled') else ('enabled',)
        if any(a.get(k) != b.get(k) for k in keys):
            raise ValueError('Protocolos de recorte incompatíveis entre origem e destino')
        _, originals, _ = evaluation_data(target, classes, split)
        check_leakage(source_records, originals)
    except (ValueError, OSError, KeyError) as exc:
        return {'state':'blocked','classes':classes,'reason':str(exc)}
    return {'state':'pending','classes':classes, **({'excluded_unverified_classes':excluded} if excluded else {})}


def cross_predict(source_run, target, runs_root=None, split='test', execution=None, on_created=None):
    from ntv_ai.pipelines.experiments import launch
    source_run = Path(source_run)
    config = load_json(source_run/'config.json')
    result = load_json(source_run/'result.json')
    if load_json(source_run/'status.json')['state'] != 'completed':
        raise ValueError('Origem precisa ser uma execução concluída')
    if isinstance(target, (str,Path)):
        target = load_json(Path(target))
    ready = pair_ready(config['dataset'], target, split)
    if ready['state'] != 'pending':
        return ready
    checked_artifact(ArtifactReference(**result['best']['checkpoint']))
    return launch('cross_predict', config, runs_root=runs_root,
                  checkpoint=result['best']['checkpoint']['path'], split=split,
                  target=target, source_run=str(source_run.resolve()), execution=execution, on_created=on_created)
