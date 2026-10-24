from copy import deepcopy

import pytest

from ntv_ai.data.groups import assign_splits, group_records, historical_links
from ntv_ai.domain import Detection, PredictionRecord
from ntv_ai.transforms.geometry import clip_box, consolidate, crop_record, nms, reproject_box, tile_windows


def test_crop_reprojection_and_classwise_nms(project):
    r = project["records"][0]
    r.group_id, r.split = "capture", "test"
    tile = crop_record(r, (5, 5, 20, 20))
    assert tile.group_id == "capture" and tile.split == "test"
    assert tile.annotations[0].geometry.coordinates == [0, 0, 9, 12]
    assert reproject_box([0, 0, 9, 12], tile.metadata["transform"]) == [5, 5, 14, 17]
    assert tile.annotations[0].flags["retained_fraction"] == .9
    items = [Detection("a", [0, 0, 10, 10], .9), Detection("a", [0, 0, 10, 10], .8),
             Detection("b", [0, 0, 10, 10], .7)]
    assert [d.class_id for d in nms(items, .5)] == ["a", "b"]
    output = consolidate([PredictionRecord(tile.image_id, [Detection("a", [0, 0, 9, 12], .9)])], [tile], .5)
    assert output[0].image_id == r.image_id
    assert output[0].detections[0].bbox_xyxy == [5, 5, 14, 17]


def test_windows_align_edges_without_padding_or_lost_last_row():
    windows = tile_windows(23, 17, 10, .2)
    assert windows[-1] == (13, 7, 23, 17)
    assert len(windows) == len(set(windows))
    assert tile_windows(4, 5, 10, 0) == [(0, 0, 4, 5)]
    assert clip_box([0, 0, 2, 2], (3, 3, 5, 5)) is None
    with pytest.raises(ValueError):
        tile_windows(20, 20, 10, 1)


def test_duplicate_group_prevents_split_leakage(project):
    records = deepcopy(project["records"])
    for r in records:
        r.group_id = r.image_id
    records[0].pixel_sha256 = records[1].pixel_sha256
    group_records(records)
    assert records[0].group_id == records[1].group_id
    issues = assign_splits(records, {"mode": "preserve"})
    assert any(i["code"] == "group_crosses_splits" for i in issues)
    assert records[0].split is None


def test_group_split_determinism_independent_of_input_order(project):
    records = deepcopy(project["records"])
    for r in records:
        r.group_id = r.image_id
    a, b = deepcopy(records), list(reversed(deepcopy(records)))
    config = {"mode": "generate", "seed": 42, "ratios": {"train": .5, "val": .25, "test": .25}}
    for batch in (a, b):
        group_records(batch)
        assert assign_splits(batch, config) == []
    assert {r.image_id: r.split for r in a} == {r.image_id: r.split for r in b}


def test_historical_links_require_current_hashes_and_ignore_absolute_paths(project):
    root = project["root"]
    a, b = project["records"][:2]
    inventory = root / "inventory.csv"
    inventory.write_text(f"dataset,image_key,sha256\nfixture,{a.path},{a.sha256}\nfixture,{b.path},{b.sha256}\n")
    review = root / "review.csv"
    review.write_text("dataset_a,image_a,dataset_b,image_b,review_status,path_a\n"
                      f"fixture,{a.path},fixture,{b.path},visually_equivalent,/obsolete/path\n")
    assert historical_links([a, b], [inventory], review) == [(a.image_id, b.image_id)]
    inventory.write_text(inventory.read_text().replace(a.sha256, "wrong"))
    with pytest.raises(ValueError, match="hash"):
        historical_links([a, b], [inventory], review)


def test_confirmed_correlated_views_are_grouped_but_candidates_are_not(project):
    root = project["root"]
    a, b = project["records"][:2]
    inventory = root / "inventory.csv"
    inventory.write_text(f"dataset,image_key,sha256\nfixture,{a.path},{a.sha256}\nfixture,{b.path},{b.sha256}\n")
    review = root / "review.csv"
    header = "dataset_a,image_a,dataset_b,image_b,review_status\n"
    row = f"fixture,{a.path},fixture,{b.path},"
    review.write_text(header + row + "not_reviewed\n")
    assert historical_links([a, b], [inventory], review) == []
    review.write_text(header + row + "visually_correlated_views\n")
    assert historical_links([a, b], [inventory], review) == [(a.image_id, b.image_id)]


def test_consolidation_rejects_missing_failed_or_duplicate_tiles(project):
    record = project["records"][0]
    with pytest.raises(ValueError, match="todas"):
        consolidate([], [record], .5)
    with pytest.raises(ValueError, match="falha"):
        consolidate([PredictionRecord(record.image_id, [], "failed")], [record], .5)
    with pytest.raises(ValueError, match="duplicados"):
        consolidate([PredictionRecord(record.image_id, []), PredictionRecord(record.image_id, [])], [record], .5)
