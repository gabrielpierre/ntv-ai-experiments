"""Views of already reviewed data; no relabeling, resplitting or new crops."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ntv_ai.data.views import eligibility
from ntv_ai.experiments.cross import load_dataset, check_leakage, pair_ready
from ntv_ai.pipelines.benchmark_data import descriptor
from ntv_ai.storage import atomic_json, digest, load_json, write_snapshot

CLASSES = ['isolador', 'espacador', 'amortecedor']


def compose(entries, training_sources, classes=CLASSES, smoke=False):
    """Only selected source train + NTV validation enter the training snapshot."""
    loaded = {key: load_dataset(value) for key, value in entries.items()}
    records, sources, original_ids, sample_ids = {}, {}, [], []
    for name in dict.fromkeys([*training_sources, 'ntv_ai']):
        manifest, rows = loaded[name]
        if not set(classes) <= set(manifest['view']['classes']):
            raise ValueError('Incompatible class coverage')
        originals = set(manifest['view']['evaluation_image_ids'])
        samples = set(manifest['view']['image_ids'])
        selected = [r for r in rows if (r.image_id in originals or r.image_id in samples)
                    and ((r.split == 'train' and name in training_sources)
                         or (r.split == 'val' and name == 'ntv_ai'))]
        for r in selected:
            if r.metadata.get('excluded'):
                raise ValueError('Excluded image in mixture')
            if r.image_id in records:
                raise ValueError('Duplicate image in mixture')
            records[r.image_id] = deepcopy(r)
            if r.image_id in originals:
                original_ids.append(r.image_id)
            if r.image_id in samples:
                sample_ids.append(r.image_id)
        for key, source in manifest['sources'].items():
            if key in sources and sources[key] != source:
                raise ValueError('Conflicting source locator')
            sources[key] = deepcopy(source)
    original_rows = [records[k] for k in original_ids]
    # Check bytes/pixels as well as reviewed grouping; classes do not cure leakage.
    check_leakage([r for r in original_rows if r.split == 'train'],
                  [r for r in original_rows if r.split == 'val'])
    failures = eligibility([records[k] for k in sample_ids], classes)
    if failures:
        raise ValueError(f'Ineligible mixture: {failures[:3]}')
    for manifest, rows in loaded.values():
        test_ids = set(manifest['view']['evaluation_image_ids'])
        check_leakage(original_rows, [r for r in rows if r.image_id in test_ids and r.split == 'test'])
    return list(records.values()), sources, original_ids, sample_ids


def prepare_workshop(config, output):
    output = Path(output)
    previous = load_json(Path(config['previous_workspace']) / 'orchestration.json')
    catalog_path = Path(previous['preparation']['catalog'])
    original = load_json(catalog_path)
    identity = digest({'config': config, 'catalog': original})
    destination = output / 'preparation' / identity[:16]
    existing = destination / 'catalog.json'
    if existing.exists():
        result = load_json(existing)
        if result['identity'] != identity:
            raise ValueError('Preparation identity changed')
        for values in result['arms'].values():
            for entry in values.values():
                load_dataset(entry)
        return result
    destination.mkdir(parents=True, exist_ok=True)
    result = {'identity': identity, 'arms': {}, 'targets': original['datasets'],
              'original_catalog': str(catalog_path), 'counts': {},
              'selection_policy': 'NTV validation for every arm; target-aware model selection'}
    for protocol in config['protocols']:
        entries = {k: original['datasets'][k][protocol] for k in ('ntv_ai', 'ptl_ai', 'stn_plad')}
        for arm, training_sources in config['arms'].items():
            rows, sources, original_ids, sample_ids = compose(entries, training_sources, config['classes'])
            selected = [r for r in rows if r.image_id in set(original_ids)]
            counts = {s: sum(r.split == s for r in selected) for s in ('train', 'val', 'test')}
            expected = {'ntv_only': 30, 'external_pool': 128, 'mixed': 158}[arm]
            if counts != {'train': expected, 'val': 8, 'test': 0}:
                raise ValueError(f'Unexpected original counts: {arm}: {counts}')
            view = deepcopy(load_dataset(entries['ntv_ai'])[0]['view'])
            view.update(id=f'workshop-{arm}-{protocol}-{identity[:12]}', classes=config['classes'],
                        image_ids=sample_ids, evaluation_image_ids=original_ids, eligible=True,
                        independence='within_capture_exploratory', selection_policy=result['selection_policy'])
            folder = destination / arm / protocol
            # Recover a preparation interrupted after writing an immutable snapshot.
            manifest = folder / 'manifest.json'
            if not manifest.exists():
                if folder.exists():
                    raise ValueError(f'Incomplete snapshot; preserve and inspect {folder}')
                manifest = write_snapshot(folder, rows, sources,
                    {'operation': 'workshop_mixture', 'identity': identity,
                     'parents': {k: v['version'] for k, v in entries.items()},
                     'train_sources': training_sources, 'validation_source': 'ntv_ai'}, [], view=view)
            entry = descriptor(manifest)
            if load_dataset(entry)[0]['view'] != view:
                raise ValueError('Existing mixture differs')
            for target in entries.values():
                ready = pair_ready(entry, target)
                if ready['state'] != 'pending':
                    raise ValueError(f'Blocked evaluation: {ready}')
            result['arms'].setdefault(arm, {})[protocol] = entry
            result['counts'][f'{arm}/{protocol}'] = {
                'originals': counts,
                'samples': dict(Counter(r.split for r in rows if r.image_id in set(sample_ids))),
                'source_samples': dict(Counter(r.dataset_id for r in rows
                    if r.image_id in set(sample_ids) and r.split == 'train'))}
    atomic_json(existing, result)
    return result
