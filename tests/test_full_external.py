from copy import deepcopy
from pathlib import Path

import pytest

from ntv_ai.domain.records import Annotation, Geometry, ImageRecord, Issue
from ntv_ai.pipelines.full_external import EXTERNAL, adapt_geometry, campaign_config, support, stratify_full_groups, blocking_source_issues


def image(name='ttpla', label='tower_lattice', target='torre', geometry=None):
    row = ImageRecord('image', name, 'image.jpg', width=100, height=80)
    row.annotations = [Annotation('box', label, geometry or Geometry('polygon', [[-1, 0], [20, 0], [20, 30]]),
                                  {}, label=target, mapping_status='mapped')]
    return row


def test_mpid_unpublished_splits_do_not_hide_missing_training_data():
    unpublished = [Issue('declared_split_unavailable', 'data.yaml', s) for s in ('val', 'test')]
    assert blocking_source_issues('mpid', unpublished) == []
    assert len(blocking_source_issues('stn_plad', unpublished)) == 2
    broken = [Issue('declared_split_unavailable', 'data.yaml', 'train'),
              Issue('missing_label_file', 'labels/image.txt', 'missing', 'image')]
    assert len(blocking_source_issues('mpid', unpublished + broken)) == 2


def test_full_campaign_keeps_historical_recipes_and_excludes_ntv():
    root = Path(__file__).resolve().parents[1]
    for name in EXTERNAL:
        cfg = campaign_config(root, name)
        assert cfg['origins'] == [name]
        assert cfg['targets_by_origin'] == {name: [name]}
        assert cfg['seeds'] == [42, 43, 44]
        assert set(cfg['recipes']) == {'full640', 'tiles1280'}
        assert sum(map(len, cfg['recipes'].values())) == 6
        assert 'epochs' not in cfg  # Preserve the historical early-stopping budget.
        assert all(Path(p).is_file() for v in cfg['recipes'].values() for p in v)
    with pytest.raises(ValueError):
        campaign_config(root, 'ntv_ai')


def test_polygon_conversion_preserves_source_and_records_clipping():
    row = image(); original = deepcopy(row.annotations[0].geometry)
    changes, exclusions = adapt_geometry([row], 'ttpla')
    assert not exclusions
    assert row.annotations[0].geometry.coordinates == [0, 0, 20, 30]
    assert changes[0]['from']['coordinates'] == original.coordinates
    assert row.annotations[0].flags['geometry_conversion']['from']['coordinates'] == original.coordinates


def test_void_is_not_silently_trained_as_negative():
    row = image()
    row.annotations.append(Annotation('void', 'void', Geometry('polygon', [[1, 1], [2, 1], [2, 2]]),
                                      {}, mapping_status='excluded', flags={'ignore': True}))
    changes, excluded = adapt_geometry([row], 'ttpla')
    assert row.metadata['excluded']
    assert len(excluded) == 1
    assert 'void_image_exclusion_as_in_historical_protocol' in excluded[0]['reasons']
    assert len(row.annotations) == 2


def test_invalid_boxes_and_unmapped_labels_do_not_enter_training():
    row = image(geometry=Geometry('bbox', [200, 0, 220, 10]))
    _, excluded = adapt_geometry([row], 'ttpla')
    assert excluded and row.metadata['excluded']
    row = image(); row.annotations[0].mapping_status = 'pending'
    with pytest.raises(ValueError, match='Unmapped'):
        adapt_geometry([row], 'ttpla')


def test_support_counts_images_not_tiles_and_retains_negatives():
    positive = image(); positive.split = 'train'; positive.group_id = 'a'
    negative = image(); negative.annotations = []; negative.split = 'train'; negative.group_id = 'b'
    report = support([positive, negative], ['torre'])
    assert report['train'] == {'images': 2, 'groups': 2, 'annotations': {'torre': 1},
                               'positive_images': {'torre': 1}, 'negative_images': 1}
    assert report['test']['annotations'] == {'torre': 0}


def test_large_split_is_complete_deterministic_and_grouped():
    rows = []
    for i in range(1000):
        row = image(geometry=Geometry('bbox', [1, 1, 10, 10]))
        row.image_id = str(i); row.group_id = f'g{i // 2}'
        row.metadata['material'] = str((i // 2) % 3)
        rows.append(row)
    second = deepcopy(rows)
    first = stratify_full_groups(rows, ['torre'])
    assert stratify_full_groups(second, ['torre']) == first
    assert [r.split for r in rows] == [r.split for r in second]
    assert first['actual'] == first['targets']
    assert sum(first['actual'].values()) == 1000
    assert first['signatures'] == 3
    assert all(rows[i].split == rows[i + 1].split for i in range(0, 1000, 2))
    with pytest.raises(ValueError, match='three positive groups'):
        stratify_full_groups(rows[:4], ['torre'])
