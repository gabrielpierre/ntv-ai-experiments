"""Full-source counterpart of filtered_data; historical inputs are never changed."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ntv_ai.data.adapters import READERS
from ntv_ai.data.harmonize import harmonize
from ntv_ai.data.stratified import split_targets, stratify_groups
from ntv_ai.data.views import eligibility
from ntv_ai.domain.records import Geometry
from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.runs import Run
from ntv_ai.pipelines.benchmark_data import descriptor, tile_impact
from ntv_ai.pipelines.filtered_data import CLASSES, evidence_groups
from ntv_ai.pipelines.prepare import materialize_tiles
from ntv_ai.storage import atomic_json, digest, file_hash, load_json, write_snapshot

EXTERNAL = ('stn_plad', 'ptl_ai', 'ttpla', 'mpid')
EXPECTED_IMAGES = {'stn_plad': 133, 'ptl_ai': 6295, 'ttpla': 1242, 'mpid': 5019}
MODELS = ('yolo26-s', 'dfine-s', 'rfdetr-nano')
PROTOCOLS = ('full640', 'tiles1280')


def stratify_full_groups(records, classes, seed=42):
    """Same count/coverage objectives, compacted by identical group signatures.

    Thousands of interchangeable binary variables made the subset solver time out.
    An integer count per signature is equivalent for the image/stratum objectives;
    a seeded order assigns the actual groups afterwards without splitting them.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    groups = defaultdict(list)
    for r in records:
        if not r.group_id:
            raise ValueError('Split requires established group identities')
        groups[r.group_id].append(r)
    materials = sorted({r.metadata['material'] for r in records if r.metadata.get('material')})
    strata = list(classes) + ['material:' + m for m in materials]
    signatures = defaultdict(list)
    for key, members in groups.items():
        counts = Counter()
        for r in members:
            counts.update({a.label for a in r.annotations if a.mapping_status == 'mapped' and a.label in classes})
            if r.metadata.get('material'):
                counts['material:' + r.metadata['material']] += 1
        signatures[tuple([len(members)] + [counts[c] for c in strata])].append(key)
    vectors = sorted(signatures)
    values = np.asarray(vectors, dtype=float)
    multiplicities = np.asarray([len(signatures[v]) for v in vectors])
    if len(groups) < 3 or any(sum(multiplicities[values[:, j] > 0]) < 3 for j in range(1, len(classes) + 1)):
        raise ValueError('At least three positive groups per class are required')
    targets = split_targets(len(records))
    desired = np.outer([30 / 46, 8 / 46, 8 / 46], (values * multiplicities[:, None]).sum(axis=0))
    desired[:, 0] = list(targets.values())
    g, d = values.shape
    variables = 3 * g + 3 * d
    rows, lows, highs = [], [], []
    def add(terms, low, high):
        rows.append(terms); lows.append(low); highs.append(high)
    for i in range(g):
        add({3 * i + s: 1 for s in range(3)}, multiplicities[i], multiplicities[i])
    for s in range(3):
        for j in range(d):
            terms = {3 * i + s: values[i, j] for i in range(g) if values[i, j]}
            slack = 3 * g + s * d + j
            add({**terms, slack: -1}, -np.inf, desired[s, j])
            add({**terms, slack: 1}, desired[s, j], np.inf)
            if 1 <= j <= len(classes):
                add(terms, 1, np.inf)
    matrix = lil_matrix((len(rows), variables))
    for i, terms in enumerate(rows):
        for j, v in terms.items():
            matrix[i, j] = v
    costs = np.zeros(variables)
    for i, vector in enumerate(vectors):
        for s in range(3):
            costs[3 * i + s] = int(digest([seed, vector, s])[:8], 16) / 2**32 / (10 * len(records))
    for s in range(3):
        costs[3 * g + s * d] = 46 * (1 + 6 * len(records) * max(1, len(strata)))
        costs[3 * g + s * d + 1:3 * g + (s + 1) * d] = 46
    result = milp(costs, integrality=np.r_[np.ones(3 * g), np.zeros(3 * d)],
                  bounds=Bounds(np.zeros(variables), np.r_[np.repeat(multiplicities, 3), np.full(3 * d, np.inf)]),
                  constraints=LinearConstraint(matrix.tocsc(), lows, highs),
                  options={'time_limit': 120, 'mip_rel_gap': 0})
    if not result.success:
        raise ValueError(f'Full-source stratification failed: {result.message}')
    for i, vector in enumerate(vectors):
        keys = sorted(signatures[vector], key=lambda key: (digest([seed, key]), key))
        assigned = 0
        for s, split in enumerate(('train', 'val', 'test')):
            count = round(result.x[3 * i + s])
            for key in keys[assigned:assigned + count]:
                for r in groups[key]:
                    r.split = split
            assigned += count
        if assigned != len(keys):
            raise ValueError('Incomplete integer assignment')
    return {'seed': seed, 'ratio': [30, 8, 8], 'targets': targets,
            'actual': dict(Counter(r.split for r in records)), 'groups': len(groups),
            'signatures': len(vectors), 'strategy': 'MILP on equivalent group signatures; seeded assignment; no sampling'}


def campaign_config(root, name):
    if name not in EXTERNAL:
        raise ValueError('Only the four external sources are permitted')
    config = read_config(Path(root) / 'configs/benchmark/filtered-campaign.yaml')
    config['origins'] = [name]
    config['targets_by_origin'] = {name: [name]}
    config['recipes'] = {p: [str(Path(root) / 'configs/benchmark' / f'{m}-{p}.yaml')
                             for m in MODELS] for p in PROTOCOLS}
    config['execution'] = {'max_concurrent_jobs': 1, 'gpu_reserve_mib': 2048, 'cpu_threads': 8}
    return config


def adapt_geometry(records, name):
    """Derived boxes only. Preserve raw annotations, ignored regions and exclusions."""
    changes, exclusions = [], []
    for r in records:
        reasons = []
        if name == 'ttpla' and any(a.original_label == 'void' for a in r.annotations):
            reasons.append('void_image_exclusion_as_in_historical_protocol')
        for a in r.annotations:
            if a.mapping_status == 'pending':
                raise ValueError(f'Unmapped native class: {name}/{a.original_label}')
            if a.flags.get('ignore') or a.flags.get('iscrowd') or a.flags.get('difficult'):
                reasons.append('unsupported_ignore_region')
            if a.label not in CLASSES[name] or a.mapping_status != 'mapped':
                continue
            original = asdict(a.geometry)
            if a.geometry.type in ('polygon', 'rectangle') and name == 'ttpla':
                points = a.geometry.coordinates
                box = [min(p[0] for p in points), min(p[1] for p in points),
                       max(p[0] for p in points), max(p[1] for p in points)]
            elif a.geometry.type == 'bbox':
                box = list(a.geometry.coordinates)
            else:
                raise ValueError(f'Unsupported target geometry: {original}')
            clipped = [max(0., min(float(r.width), box[0])), max(0., min(float(r.height), box[1])),
                       max(0., min(float(r.width), box[2])), max(0., min(float(r.height), box[3]))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                reasons.append('nonpositive_box_after_clipping')
                continue
            converted = Geometry('bbox', clipped)
            if asdict(converted) != original:
                a.flags['geometry_conversion'] = {'from': original, 'rule': 'axis_aligned_envelope_clip_to_image'}
                changes.append({'image_id': r.image_id, 'annotation_id': a.annotation_id,
                                'from': original, 'to': asdict(converted)})
                a.geometry = converted
        if reasons:
            r.metadata.update(excluded=True, exclusion_reason=sorted(set(reasons)))
            exclusions.append({'image_id': r.image_id, 'path': r.path, 'sha256': r.sha256,
                               'reasons': sorted(set(reasons))})
    return changes, exclusions


def support(records, classes):
    report = {}
    for split in ('train', 'val', 'test'):
        rows = [r for r in records if r.split == split]
        boxes = Counter(a.label for r in rows for a in r.annotations
                        if a.mapping_status == 'mapped' and a.label in classes)
        positives = Counter(c for r in rows for c in {a.label for a in r.annotations
                            if a.mapping_status == 'mapped' and a.label in classes})
        report[split] = {'images': len(rows), 'groups': len({r.group_id for r in rows}),
                         'annotations': {c: boxes[c] for c in classes},
                         'positive_images': {c: positives[c] for c in classes},
                         'negative_images': sum(not any(a.mapping_status == 'mapped' and a.label in classes
                                                       for a in r.annotations) for r in rows)}
    return report


def blocking_source_issues(name, issues):
    """Classify source findings after checking download completion and full count.

    The published MPID Drive inventory has only train directories. Its YAMLs
    also reference unpublished val/test directories. Keep these findings in the
    audit, but generate this campaign's own grouped splits from all 5,019 images.
    Missing images/labels and missing declared training directories still block.
    """
    return [asdict(i) for i in issues
            if i.code not in {'image_without_annotations', 'missing_source_split'}
            and not (name == 'mpid' and i.code == 'declared_split_unavailable'
                     and i.message in {'val', 'test'} and i.image_id is None)]


def prepare_full_external(root, sources_root, workspace, name, workers=8):
    root, sources_root, workspace = map(lambda p: Path(p).resolve(), (root, sources_root, workspace))
    if name not in EXTERNAL:
        raise ValueError('NTV and other sources are outside this campaign')
    source_root = sources_root / name
    downloaded = load_json(source_root / 'metadata/download-status.json')
    if downloaded['status'] != 'completed' or downloaded.get('errors'):
        raise ValueError(f'Full source download has not completed: {name}')
    cfg = read_config(root / f'configs/datasets/{name}.yaml')
    cfg.update(root=str(source_root), workers=workers)
    mapping = read_config(root / 'configs/benchmark/mappings.yaml')
    # 28 native labels use this capitalization in the full TTPLA release.
    # Keep original_label intact and record the explicit alias in resolved-mapping.
    mapping['datasets']['ttpla']['Cable'] = deepcopy(mapping['datasets']['ttpla']['cable'])
    mapping['datasets']['ttpla']['tower_tucohy'] = {
        'exclude': True, 'reason': 'Outside the NTV lattice-tower task, as in filtered campaign'}
    taxonomy = read_config(root / 'configs/benchmark/taxonomy.yaml')
    with Run(workspace / 'preparation' / name, 'full_external_prepare', cfg) as run:
        print(f'{name}: reading full source', flush=True)
        reader = READERS[cfg['format']](cfg, source_root)
        records = reader.read()
        atomic_json(run.path / 'source-issues.json', [asdict(i) for i in reader.issues])
        if len(records) != EXPECTED_IMAGES[name]:
            raise ValueError(f'{name}: expected full inventory {EXPECTED_IMAGES[name]}, received {len(records)}')
        fatal = blocking_source_issues(name, reader.issues)
        if fatal:
            raise ValueError(f'{name}: unresolved source issues: {fatal[:5]}')
        raw_count = sum(len(r.annotations) for r in records)
        source = {'root': str(source_root), 'evidence': reader.evidence, 'config': cfg,
                  'version': digest({r.path: r.sha256 for r in records} | reader.evidence)}
        records = harmonize(records, taxonomy, mapping)
        changes, exclusions = adapt_geometry(records, name)
        atomic_json(run.path / 'geometry-changes.json', changes)
        atomic_json(run.path / 'exclusions.json', exclusions)
        atomic_json(run.path / 'resolved-mapping.json', mapping)
        history = evidence_groups(records, root / 'metadata/external_datasets/v1')
        for r in records:
            r.annotation_scope = dict.fromkeys(CLASSES[name], 'exhaustive')
            r.metadata['coverage_basis'] = 'Native published detection labels; no claim of new manual review'
            r.metadata['grouping_basis'] = 'Exact file/pixel identity, named source variants, existing historical links'
        active = [r for r in records if not r.metadata.get('excluded')]
        print(f'{name}: splitting {len(active)} eligible images; {len(exclusions)} exclusions', flush=True)
        split_report = (stratify_full_groups if len(active) > 500 else stratify_groups)(active, CLASSES[name], 42)
        split_report['distribution'] = support(active, CLASSES[name])
        memberships = defaultdict(set)
        for r in active:
            memberships[r.group_id].add(r.split)
        if any(len(v) != 1 for v in memberships.values()):
            raise ValueError('Related images cross partitions')
        problems = eligibility(active, CLASSES[name])
        atomic_json(run.path / 'eligibility.json', problems)
        if problems:
            raise ValueError(f'{name}: eligibility failures: {problems[:5]}')
        atomic_json(run.path / 'splits-report.json', split_report)
        atomic_json(run.path / 'members.json', [
            {'image_id': r.image_id, 'path': r.path, 'sha256': r.sha256, 'pixel_sha256': r.pixel_sha256,
             'group_id': r.group_id, 'split': r.split, 'original_splits': r.original_splits,
             'excluded': r.metadata.get('excluded', False)} for r in records])
        lineage = {'operation': 'full_external_prepare', 'mapping_sha256': digest(mapping),
                   'taxonomy_sha256': digest(taxonomy), 'historical_evidence': history,
                   'split_policy': 'Same as filtered campaign: grouped 30:8:8, seed 42; native splits retained as metadata',
                   'native_annotations': True, 'manually_corrected_subsets_added': False}
        catalog = {'purpose': 'exploratory', 'datasets': {name: {}}, 'storage_root': str(root),
                   'preparation_run': run.id}
        output = workspace / 'prepared' / name / run.id
        for protocol in PROTOCOLS:
            print(f'{name}: preparing {protocol}', flush=True)
            transform = {'enabled': protocol == 'tiles1280', 'size': 1280, 'overlap': .25, 'border_policy': 'clip'}
            sources = {name: deepcopy(source)}
            stored, samples = deepcopy(records), active
            if transform['enabled']:
                samples = materialize_tiles(active, sources, CLASSES[name], transform, output / 'tiles', workers)
                stored.extend(samples)
            view = {'id': f'{name}-full-source-{protocol}-v1', 'classes': CLASSES[name],
                    'image_ids': [r.image_id for r in samples], 'evaluation_image_ids': [r.image_id for r in active],
                    'eligible': True, 'purpose': 'exploratory', 'splits': {'mode': 'explicit'}, 'tiling': transform,
                    'independence': 'grouped_published_source; not official split scores'}
            folder = output / protocol
            manifest = write_snapshot(folder, stored, sources, lineage, [], view=view)
            atomic_json(folder / 'splits.json', {s: [r.image_id for r in active if r.split == s]
                                               for s in ('train', 'val', 'test')})
            entry = {**descriptor(manifest), 'eligible': True, 'blockers': [],
                     'counts': dict(Counter(r.split for r in active))}
            catalog['datasets'][name][protocol] = entry
            atomic_json(folder / 'dataset.json', descriptor(manifest))
            if transform['enabled']:
                atomic_json(folder / 'tile-impact.json', tile_impact(active, samples, CLASSES[name]))
        atomic_json(run.path / 'datasets.json', catalog)
        result = {'dataset': name, 'run_id': run.id, 'catalog': str(run.path / 'datasets.json'),
                  'source_images': len(records), 'source_annotations': raw_count,
                  'eligible_images': len(active), 'excluded_images': len(exclusions),
                  'splits': split_report, 'source_version': source['version']}
        atomic_json(run.path / 'result.json', result)
        atomic_json(workspace / 'ready' / f'{name}.json', result)
        print(f'{name}: ready {result["catalog"]}', flush=True)
        return result
