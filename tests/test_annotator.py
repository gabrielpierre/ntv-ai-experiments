from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import random

import pytest

from ntv_ai.annotator.data import authoritative_union, protected_hashes
from ntv_ai.annotator.lofo import audit_sample, check_validation
from ntv_ai.annotator.selection import choose, operating_point
from ntv_ai.annotator.splits import grouped_split
from ntv_ai.domain.records import Annotation, Geometry, ImageRecord


def record(identifier, label="torre", **metadata):
    return ImageRecord(str(identifier), "fixture", f"{identifier}.jpg", 100, 100,
                       sha256=str(identifier), pixel_sha256=f"pixels-{identifier}",
                       annotations=[Annotation(f"box-{identifier}", label, Geometry("bbox", [0, 0, 10, 10]),
                                               {}, label=label, mapping_status="mapped")], metadata=metadata)


def test_reviewed_version_replaces_boxes_and_pixel_duplicates():
    old = record("old", capture_id="001")
    new = record("new", capture_id="001"); new.pixel_sha256 = old.pixel_sha256
    new.annotations[0].geometry.coordinates = [1, 2, 8, 9]
    rows, duplicates = authoritative_union([(old, 0), (new, 1)])
    assert len(rows) == 1 and len(rows[0].annotations) == 1
    assert rows[0].annotations[0].geometry.coordinates == [1, 2, 8, 9]
    assert duplicates[0]["kept_priority"] == 1
    assert old.annotations[0].geometry.coordinates == [0, 0, 10, 10]


def test_pending_edits_are_protected_but_not_untouched_proposals():
    coco = {"images": [{"id": i, "sha256": str(i)} for i in range(3)]}
    states = [{"image_id": 0, "status": "accepted", "revision": 1},
              {"image_id": 1, "status": "pending", "revision": 1},
              {"image_id": 2, "status": "pending", "revision": 0}]
    assert protected_hashes(coco, {"states": states}) == {"0", "1"}


def test_group_partition_is_repeatable_and_respects_visual_edges():
    pytest.importorskip("scipy")
    rng = random.Random(17); rows = []; anchors = {}
    for c in ("001", "002", "003"):
        anchors[c] = "2025:09:11 10:00:00"
        for i in range(5):
            time = datetime(2025, 9, 11, 10) + timedelta(minutes=3*i)
            rows.append(record(c+str(i), capture_id=c, datetime=time.strftime("%Y:%m:%d %H:%M:%S"),
                               phash=rng.getrandbits(64), dhash=rng.getrandbits(64)))
    rows[1].metadata.update(phash=rows[0].metadata["phash"], dhash=rows[0].metadata["dhash"])
    options = {"min_objects": 1, "min_positive_groups": 1, "seed": 43}
    a = grouped_split(deepcopy(rows), anchors, ["torre"], options)
    b = grouped_split(list(reversed(deepcopy(rows))), anchors, ["torre"], options)
    assert a["assignment"] == b["assignment"]
    assert set(a["assignment"].values()) <= {"train", "val", "buffer"}
    assigned = {a["assignment"][r.sha256] for r in rows[:2]} - {"buffer"}
    assert len(assigned) <= 1
    c = grouped_split(deepcopy(rows), anchors, ["torre"], options, a["assignment"])
    assert c["assignment"] == a["assignment"]


def test_f2_matching_does_not_double_count_or_require_precision_floor():
    r = record("1")
    predictions = {"1": [{"class_id": "torre", "bbox_xyxy": [0, 0, 10, 10], "score": .8},
                          {"class_id": "torre", "bbox_xyxy": [0, 0, 10, 10], "score": .7},
                          {"class_id": "torre", "bbox_xyxy": [20, 20, 30, 30], "score": .6}]}
    point = operating_point([r], predictions, "torre", .1)
    assert (point["tp"], point["fp"], point["fn"]) == (1, 2, 0)
    assert point["precision"] < .5 and point["recall"] == 1
    assert choose([{**point, "candidate": "a"}])["candidate"] == "a"
    assert operating_point([r], predictions, "isolador", .1)["f2"] is None


def test_final_benchmark_rejects_legacy_approval_without_second_person():
    r = record("1", review_stage="reviewed")
    with pytest.raises(ValueError, match="second validation"):
        check_validation([r])
    r.metadata.update(review_stage="validated", review_evidence={"review_stage": "validated", "reviewed_by": "a", "validated_by": "a"})
    with pytest.raises(ValueError, match="distinct"):
        check_validation([r])


def test_audit_sample_covers_rare_class_and_negatives():
    rows = [record(str(i)) for i in range(100)]
    rows[4].annotations[0].label = "sinalizacao"
    rows[7].annotations = []
    sample = audit_sample(rows)
    assert len(sample) == 10 and "4" in sample and "7" in sample


def test_validation_is_bound_to_actual_geometry():
    r = record("1", review_stage="validated")
    r.annotations[0].original = {"id": "p-1", "bbox": [0, 0, 10, 10]}
    fingerprint = hashlib.sha256(json.dumps([("p-1", 1, [0., 0., 10., 10.])], separators=(",", ":")).encode()).hexdigest()
    r.metadata["review_evidence"] = {"review_stage": "validated", "reviewed_by": "first", "validated_by": "second",
        "validated_at": "2026-10-19", "reviewed_resolution": "original", "validated_annotation_sha256": fingerprint}
    check_validation([r])
    r.annotations[0].geometry.coordinates[2] = 11
    with pytest.raises(ValueError, match="Geometry differs"):
        check_validation([r])


def test_lofo_preflight_blocks_missing_audit_and_annotator_weights(tmp_path):
    from ntv_ai.annotator.lofo import preflight_guard
    from ntv_ai.storage import atomic_json, file_hash
    official = tmp_path/"official.pt"; official.write_bytes(b"official")
    ref = {"path": str(official), "sha256": file_hash(official)}
    atomic_json(tmp_path/"configs/experiments/models/rfdetr-nano.yaml", {"weights": ref})
    config = {"runtime": {"storage_root": str(tmp_path)}, "model": {"backend": "rfdetr", "weights": ref}}
    manifest = {"lineage": {"operation": "lofo"}}
    with pytest.raises(ValueError, match="audit"):
        preflight_guard(config, manifest, tmp_path/"fold/full/manifest.json")
    config["model"]["weights"] = {"path": str(official), "sha256": "annotator"}
    with pytest.raises(ValueError, match="official initialization"):
        preflight_guard(config, manifest, tmp_path/"fold/full/manifest.json")
