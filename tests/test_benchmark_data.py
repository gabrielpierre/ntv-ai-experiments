from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from PIL import Image

from ntv_ai.data.adapters.readers import YoloReader
from ntv_ai.data.harmonize import harmonize
from ntv_ai.domain.records import ImageRecord
from ntv_ai.pipelines.benchmark_data import temporal_pilot


def test_separate_yolo_layout_preserves_received_files(tmp_path):
    (tmp_path / 'images').mkdir()
    labels = tmp_path / 'export/labels/train'
    labels.mkdir(parents=True)
    Image.new('RGB', (100, 80)).save(tmp_path / 'images/a.png')
    (labels / 'a.txt').write_text('4 0.5 0.5 0.2 0.4\n')
    (tmp_path / 'export/train.txt').write_text('old/root/images/train/a.png\n')
    data = tmp_path / 'export/data.yaml'
    data.write_text('names: {4: isolador}\ntrain: train.txt\npath: .\n')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    spec = {'id': 'fixture', 'images': ['images/*'], 'subsets': [
        {'name': 'new', 'yaml': 'export/data.yaml', 'layout': {'images': 'images', 'labels': 'export/labels/train'}}]}
    reader = YoloReader(spec, tmp_path)
    rows = reader.read()
    assert not reader.issues
    assert len(rows) == 1 and rows[0].annotations[0].original_label == 'isolador'
    assert rows[0].annotations[0].geometry.coordinates == [40, 24, 60, 56]
    assert rows[0].original_splits[0]['split'] == 'train'
    assert all(p.read_bytes() == value for p,value in before.items())
    assert 'export/train.txt' in reader.evidence


def records():
    return [ImageRecord(str(i), 'ntv_ai', f'raw/images/{i:03d}.jpg', sha256=f'bytes-{i}', pixel_sha256=f'pixels-{i}') for i in range(50)]


def test_temporal_pilot_is_deterministic_and_keeps_embargo():
    a = temporal_pilot(records())
    b = temporal_pilot(list(reversed(records())))
    assert a == b
    assert [sum(r.split == s for r in a) for s in ('train','val','test',None)] == [30,8,8,4]
    assert all(r.metadata['independence'] == 'within_capture_exploratory' for r in a)
    assert [r.metadata['temporal_index'] for r in a if r.metadata['excluded']] == [30,31,40,41]


def test_temporal_pilot_rejects_duplicate_across_partitions():
    rows = records(); rows[42].pixel_sha256 = rows[0].pixel_sha256
    with pytest.raises(ValueError, match='Duplicata'):
        temporal_pilot(rows)


def test_benchmark_mapping_preserves_foundations_and_unifies_signage(project):
    root = Path(__file__).resolve().parents[1]
    tax = yaml.safe_load((root/'configs/benchmark/taxonomy.yaml').read_text())
    mapping = yaml.safe_load((root/'configs/benchmark/mappings.yaml').read_text())
    row = deepcopy(project['records'][0]); row.dataset_id = 'ntv_ai'
    original = row.annotations[0]; row.annotations = []
    for label in ['torre_trelica','placa_id','placa_sinalizacao','fundacao_estai','fundacao_concreto']:
        a = deepcopy(original); a.original_label = label; row.annotations.append(a)
    actual = harmonize([row], tax, mapping)[0]
    assert [a.label for a in actual.annotations] == ['torre','sinalizacao','sinalizacao','fundacao_estai','fundacao_concreto']
    assert actual.annotations[0].original_label == 'torre_trelica'


def test_tiles_report_large_objects_and_keep_negative_crops(project):
    from ntv_ai.pipelines.benchmark_data import tile_impact
    from ntv_ai.transforms.geometry import crop_record, tile_windows
    r=deepcopy(project['records'][0]); r.width=5000; r.height=4000; r.split='train'
    a=r.annotations[0]; a.label='torre'; a.mapping_status='mapped'; a.geometry.coordinates=[0,0,5000,2000]
    tiles=[crop_record(r,w) for w in tile_windows(r.width,r.height,1280,.25)]
    report=tile_impact([r],tiles,['torre'])
    assert report['by_class']['torre']['cannot_reach_iou50']==1
    assert report['negative_tiles']>0
    assert all(t.split=='train' for t in tiles)
