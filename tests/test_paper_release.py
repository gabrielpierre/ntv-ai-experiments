"""Release contracts against the frozen paper evidence (CPU only)."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import yaml

from ntv_ai.cli import execute, parser
from ntv_ai.storage import read_snapshot, load_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_release', ROOT/'scripts/paper.py')
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def view(name):
    index = load_json(ROOT/'paper/datasets/index.json')
    key = next(k for k, v in index.items() if v['view_id'] == name)
    return paper.frozen(key)


def originals(manifest, records, split):
    ids = set(manifest['view']['evaluation_image_ids'])
    return [r for r in records if r.image_id in ids and r.split == split]


def test_main_test_retains_original_eight_without_training_overlap():
    old, old_rows = view('ntv_ai-filtered-full640-v1')
    new, new_rows = view('expanded18')
    test = originals(new, new_rows, 'test')
    assert len(test) == 18
    assert sum(len(r.annotations) for r in test) == 411
    assert {r.sha256 for r in originals(old, old_rows, 'test')} < {r.sha256 for r in test}
    trainval = [r for r in old_rows if r.split in ('train', 'val')]
    assert not {r.sha256 for r in test} & {r.sha256 for r in trainval}
    assert not {r.pixel_sha256 for r in test} & {r.pixel_sha256 for r in trainval}
    assert len(originals(old, old_rows, 'train')) == 30
    assert len(originals(old, old_rows, 'val')) == 8


def test_paper_arms_and_evaluation_counts():
    catalog = paper.catalog()
    assert Counter(x['study'] for x in catalog['training']) == {
        'main': 18, 'external': 72, 'mixed': 18, 'annotation': 1}
    for study, count in [('main', 18), ('external', 144), ('mixed', 54)]:
        assert sum(len(x['targets']) for x in catalog['training'] if x['study'] == study) == count
    assert Counter(x['split'] for x in catalog['vlm']) == {'val': 16, 'test': 5}
    for x in catalog['training']:
        assert x['seed'] == x['config']['training']['seed']
        assert x['config']['model']['weights']['path'].startswith('data/assets/')
        assert len(x['config']['model']['weights']['sha256']) == 64


def test_mixed_arms_share_ntv_validation():
    validation = []
    for arm, count in [('ntv_only', 30), ('external_pool', 128), ('mixed', 158)]:
        m, rows = view(f'workshop-{arm}-full640-32ebec166da9')
        assert len(originals(m, rows, 'train')) == count
        assert m['view']['classes'] == ['isolador', 'espacador', 'amortecedor']
        validation.append({r.sha256 for r in originals(m, rows, 'val')})
    assert len(validation[0]) == 8
    assert validation[0] == validation[1] == validation[2]


@pytest.mark.parametrize('protocol', ['full640', 'tiles1280'])
def test_portable_preparation_preserves_originals_and_resumes(project, tmp_path, monkeypatch, protocol):
    # This test exercises relocation, not the Linux-only YOLO symlink exporter.
    project['config']['exports'] = ['coco']
    project['prepare'].write_text(yaml.safe_dump(project['config']), encoding='utf-8')
    result = execute(parser().parse_args(['prepare', '--config', str(project['prepare'])]))
    manifest, rows = read_snapshot(Path(result['manifest']))
    monkeypatch.setattr(paper, 'frozen', lambda _: (deepcopy(manifest), deepcopy(rows)))
    roots = {k: v['root'] for k, v in manifest['sources'].items()}
    original_hashes = {p: file_hash(p) for p in project['raw'].rglob('*.png')}
    output = tmp_path/'portable'
    entry = paper.prepare_one('fixture', protocol, roots, output)
    m, prepared = read_snapshot(Path(entry['manifest']['path']))
    assert len(m['view']['evaluation_image_ids']) == 3
    assert Counter(r.split for r in prepared if not r.metadata.get('transform')) == Counter(r.split for r in rows)
    assert paper.prepare_one('fixture', protocol, roots, output) == entry
    assert original_hashes == {p: file_hash(p) for p in original_hashes}
    corrupt = next(iter(original_hashes))
    corrupt.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Missing or different image'):
        paper.prepare_one('fixture', protocol, roots, output)


def test_replay_detects_a_changed_metric():
    with pytest.raises(AssertionError):
        paper.close_metrics({'AP50': .52}, {'AP50': .53}, 'fixture')
