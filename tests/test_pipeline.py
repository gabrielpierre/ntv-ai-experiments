import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from ntv_ai.cli import execute, main, parser
from ntv_ai.data.audit import audit_records
from ntv_ai.data.harmonize import curate, harmonize
from ntv_ai.data.views import eligibility
from ntv_ai.domain import Detection, PredictionRecord
from ntv_ai.evaluation.bbox import evaluate
from ntv_ai.storage import atomic_json, file_hash, read_snapshot, safe_output, write_snapshot

from conftest import save_json, save_yaml


def run_preparation(project):
    result = execute(parser().parse_args(["prepare", "--config", str(project["prepare"])]))
    manifest, records = read_snapshot(Path(result["manifest"]))
    return result, manifest, records


def test_end_to_end_exports_and_repeatable_evaluation(project):
    hashes = {str(p): file_hash(p) for p in project["raw"].rglob("*") if p.is_file()}
    result, manifest, records = run_preparation(project)
    assert result["eligible"]
    assert {r.split for r in records} == {"train", "val", "test"}
    assert all(r.annotations[0].attributes["material"] == "unknown" for r in records)
    output = Path(result["manifest"]).parent
    coco = json.loads((output / "exports/coco/test.json").read_text())
    assert coco["annotations"][0]["bbox"] == [4, 5, 10, 12]
    assert coco["categories"] == [{"id": 1, "name": "isolador"}]
    test = next(r for r in records if r.split == "test")
    assert (output / f"exports/yolo/images/test/{test.image_id}.png").is_symlink()
    numbers = (output / f"exports/yolo/labels/test/{test.image_id}.txt").read_text().split()
    assert numbers[0] == "0"
    assert list(map(float, numbers[1:])) == pytest.approx([0.225, 11/30, .25, .4])
    predictions = project["root"] / "predictions.jsonl"
    predictions.write_text(json.dumps(asdict(PredictionRecord(test.image_id, [Detection("isolador", [4, 5, 14, 17], .9)])))+"\n")
    metrics = evaluate(manifest, records, predictions, {"split": "test"})
    assert metrics["mAP50_95"] == pytest.approx(1)
    assert metrics["per_class"]["isolador"]["AP75"] == pytest.approx(1)
    assert evaluate(manifest, records, predictions, {"split": "test"}) == metrics
    evaluation_config = save_yaml(project["root"] / "evaluate.yaml", {"manifest": result["manifest"], "split": "test", "runs": "runs"})
    response = execute(parser().parse_args(["evaluate", "--config", str(evaluation_config), "--predictions", str(predictions)]))
    config = json.loads((project["root"] / "runs" / response["run_id"] / "config.json").read_text())
    assert config["predictions"] == str(predictions)
    assert hashes == {str(p): file_hash(p) for p in project["raw"].rglob("*") if p.is_file()}


def test_empty_wrong_and_missing_predictions(project):
    _, manifest, records = run_preparation(project)
    test = next(r for r in records if r.split == "test")
    path = project["root"] / "predictions.jsonl"
    for detections in ([], [Detection("isolador", [25, 20, 35, 28], .9)]):
        path.write_text(json.dumps(asdict(PredictionRecord(test.image_id, detections))))
        assert evaluate(manifest, records, path, {"split": "test"})["AP50"] == 0
    path.write_text("")
    with pytest.raises(ValueError, match="ausentes"):
        evaluate(manifest, records, path, {"split": "test"})
    path.write_text(json.dumps({"image_id": test.image_id, "status": "failed", "detections": []}))
    with pytest.raises(ValueError, match="falha"):
        evaluate(manifest, records, path, {"split": "test"})


def test_partial_supervision_blocks_exports_and_records_failure(project):
    decisions = json.loads((project["root"] / "curation.json").read_text())
    r = project["records"][0]
    decisions["decisions"].append({"image_id": r.image_id, "image_sha256": r.sha256,
                                  "action": "set_scope", "scope": {"isolador": "partial"}, "reason": "Incomplete labels"})
    save_json(project["root"] / "curation.json", decisions)
    assert main(["prepare", "--config", str(project["prepare"])]) == 1
    statuses = [json.loads(p.read_text()) for p in (project["root"] / "runs").glob("*/status.json")]
    failed = next(s for s in statuses if s["state"] == "failed")
    assert failed["operation"] == "prepare"
    manifest_path = Path(failed["artifacts"]["manifest"])
    assert not read_snapshot(manifest_path)[0]["training_ready"]
    assert not (manifest_path.parent / "exports").exists()


def test_canonical_artifacts_are_immutable_and_detect_tampering(project):
    manifest, records = read_snapshot(project["manifest"])
    with pytest.raises(FileExistsError):
        write_snapshot(project["manifest"].parent, records, manifest["sources"], {}, [])
    with pytest.raises(FileExistsError):
        atomic_json(project["manifest"], {})
    path = project["manifest"].parent / "images.jsonl"
    path.write_text(path.read_text().replace('"width": 40', '"width": 41'))
    with pytest.raises(ValueError, match="hash"):
        read_snapshot(project["manifest"])


def test_source_changes_and_output_in_raw_are_rejected(project):
    manifest, records = read_snapshot(project["manifest"])
    image_path = project["raw"] / records[0].path
    image_path.write_bytes(image_path.read_bytes() + b"changed")
    report = audit_records(manifest, records)
    assert "image_integrity_mismatch" in report["issue_counts"]
    with pytest.raises(ValueError, match="alteradas"):
        run_preparation(project)
    with pytest.raises(ValueError, match="protegida"):
        safe_output(project["raw"] / "generated", [project["raw"]])
    link = project["root"] / "alias"
    link.symlink_to(project["raw"], target_is_directory=True)
    with pytest.raises(ValueError, match="protegida"):
        safe_output(link / "generated", [project["raw"]])


def test_unknown_mapping_and_unverified_convention_block_view(project):
    _, manifest, records = run_preparation(project)
    records[0].annotations[0].mapping_status = "pending"
    assert "mapping_pending" in eligibility(records, ["isolador"])[0]["codes"]
    records[0].annotations[0].mapping_status = "mapped"
    records[0].annotations[0].geometry.convention = "unverified_native_endpoints"
    assert "coordinate_convention_unverified" in eligibility(records, ["isolador"])[0]["codes"]


def test_curation_is_bound_to_version_and_hash(project):
    manifest, records = read_snapshot(project["manifest"])
    with pytest.raises(ValueError, match="versão"):
        curate(records, {"dataset_version": "wrong"}, manifest["version"])
    entry = {"image_id": records[0].image_id, "image_sha256": "wrong", "reason": "review", "action": "exclude_image"}
    with pytest.raises(ValueError, match="hash"):
        curate(records, {"dataset_version": manifest["version"], "decisions": [entry]}, manifest["version"])


def test_generic_components_and_material_context_are_not_invented(project):
    records = project["records"]
    records[0].annotations[0].original_label = "generic_spacer"
    tax = {"labels": {}, "coarse_labels": {"espacador": {"attributes": {}}}}
    mapped = harmonize(records, tax, {"datasets": {"fixture": {"generic_spacer": {"target": "espacador"}}}})
    assert mapped[0].annotations[0].label == "espacador"
    assert mapped[0].annotations[0].attributes == {}
    assert records[0].annotations[0].label is None


def test_importing_package_has_no_side_effects(tmp_path):
    script = ("import ntv_ai, ntv_ai.cli, ntv_ai.domain.contracts; "
              "from pathlib import Path; import sys; "
              "assert not list(Path('.').iterdir()); "
              "assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules")
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


def test_materialized_tiles_keep_parents_and_original_evaluation_view(project):
    config = project["config"]
    config["tiling"] = {"enabled": True, "size": 20, "overlap": .25, "border_policy": "clip"}
    save_yaml(project["prepare"], config)
    result, manifest, records = run_preparation(project)
    parents = [r for r in records if not r.metadata.get("transform")]
    tiles = [r for r in records if r.metadata.get("transform")]
    assert len(parents) == 3 and len(tiles) > len(parents)
    by_id = {r.image_id: r for r in parents}
    for tile in tiles:
        parent = by_id[tile.metadata["transform"]["parent_image_id"]]
        assert (tile.split, tile.group_id) == (parent.split, parent.group_id)
        assert Path(manifest["sources"][tile.dataset_id]["root"], tile.path).is_file()
    assert set(manifest["view"]["evaluation_image_ids"]) == set(by_id)
    assert not audit_records(manifest, records)["issues"]
    test = next(r for r in parents if r.split == "test")
    predictions = project["root"] / "global.jsonl"
    predictions.write_text(json.dumps(asdict(PredictionRecord(test.image_id, [Detection("isolador", [4, 5, 14, 17], 1)]))))
    assert evaluate(manifest, records, predictions, {"split": "test"})["AP50"] == pytest.approx(1)


@pytest.mark.parametrize("detection", [
    Detection("unknown", [4, 5, 14, 17], .9),
    Detection("isolador", [4, 5, 14, 17], float("nan")),
    Detection("isolador", [-1, 5, 14, 17], .9),
    Detection("isolador", [4, 5, 4, 17], .9),
])
def test_evaluation_rejects_invalid_predictions(project, detection):
    _, manifest, records = run_preparation(project)
    record = next(r for r in records if r.split == "test")
    path = project["root"] / "invalid.jsonl"
    path.write_text(json.dumps(asdict(PredictionRecord(record.image_id, [detection]))))
    with pytest.raises(ValueError):
        evaluate(manifest, records, path, {"split": "test"})


def test_annotation_exclusion_cannot_create_a_negative(project):
    _, manifest, records = run_preparation(project)
    record = records[0]
    curate(records, {"dataset_version": manifest["version"], "decisions": [
        {"action": "exclude_annotation", "image_id": record.image_id, "image_sha256": record.sha256,
         "annotation_id": record.annotations[0].annotation_id, "reason": "Requires review"}
    ]}, manifest["version"])
    assert "annotation_exclusion_requires_image_review" in eligibility(records, ["isolador"])[0]["codes"]


def test_explicit_geometry_correction_preserves_native_annotation(project):
    _, manifest, records = run_preparation(project)
    record = records[0]
    original = dict(record.annotations[0].original)
    curate(records, {"dataset_version": manifest["version"], "decisions": [
        {"action": "replace_geometry", "image_id": record.image_id, "image_sha256": record.sha256,
         "annotation_id": record.annotations[0].annotation_id, "reason": "Reviewed derived correction",
         "geometry": {"type": "bbox", "coordinates": [1, 2, 8, 9]}}
    ]}, manifest["version"])
    assert record.annotations[0].original == original
    assert record.annotations[0].geometry.coordinates == [1, 2, 8, 9]
    assert eligibility(records, ["isolador"]) == []


def test_material_mapping_uses_source_context(project):
    records = project["records"]
    for r in records:
        r.metadata["material"] = "vidro"
    taxonomy = {"labels": {"isolador": {"attributes": {"material": ["vidro", "polimero"]}}}}
    mapping = {"datasets": {"fixture": {"native": {"target": "isolador", "context_attributes": {"material": "material"}}}}}
    mapped = harmonize(records, taxonomy, mapping)
    assert all(r.annotations[0].attributes["material"] == "vidro" for r in mapped)
    records[0].metadata.pop("material")
    assert harmonize(records, taxonomy, mapping)[0].annotations[0].attributes["material"] == "unknown"


def test_reconciliation_reports_differences_without_changing_counts(project):
    from ntv_ai.data.audit import reconcile_counts
    manifest, records = read_snapshot(project["manifest"])
    report = audit_records(manifest, records)
    baseline = project["root"] / "baseline.csv"
    baseline.write_text("dataset,images,annotations\nfixture,3,3\n")
    assert reconcile_counts(report, baseline)["matches"]
    baseline.write_text("dataset,images,annotations\nfixture,3,9\n")
    result = reconcile_counts(report, baseline)
    assert not result["matches"]
    assert result["datasets"]["fixture"]["actual"]["annotations"] == 3


def test_evaluation_does_not_need_original_images_or_model(project):
    _, manifest, records = run_preparation(project)
    test = next(r for r in records if r.split == "test")
    path = project["root"] / "empty.jsonl"
    path.write_text(json.dumps(asdict(PredictionRecord(test.image_id, []))))
    for image in project["raw"].rglob("*.png"):
        image.unlink()
    assert evaluate(manifest, records, path, {"split": "test"})["AP50"] == 0
