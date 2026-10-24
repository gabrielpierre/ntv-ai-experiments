import json
from copy import deepcopy
from pathlib import Path

import pytest

from conftest import save_yaml
from ntv_ai.cli import execute, main, parser
from ntv_ai.data.eligibility import build_report
from ntv_ai.data.views import bbox_errors, eligibility
from ntv_ai.experiments.config import read_config
from ntv_ai.pipelines.eligibility import write_reports
from ntv_ai.storage import file_hash, load_json, read_snapshot, write_snapshot


@pytest.fixture
def prepared(project):
    taxonomy_path = project["root"] / "taxonomy.yaml"
    taxonomy = read_config(taxonomy_path)
    taxonomy["labels"]["cabo"] = {"attributes": {"tipo": ["condutor"]}}
    taxonomy["coarse_labels"] = {"torre": {"attributes": {}}}
    taxonomy["reference"] = "doc/reference-not-required.md"
    save_yaml(taxonomy_path, taxonomy)
    result = execute(parser().parse_args(["prepare", "--config", str(project["prepare"])]))
    config = save_yaml(project["root"] / "eligibility.yaml", {
        "taxonomy": "taxonomy.yaml", "mapping": "mapping.yaml", "runs": "reports",
        "reference_dataset": "ntv_ai",
    })
    return {**project, "prepared": Path(result["manifest"]), "report_config": config, "taxonomy": taxonomy}


def run_report(case, path=None, extra=()):
    result = execute(parser().parse_args([
        "eligibility", "--config", str(case["report_config"]),
        "--manifest", str(path or case["prepared"]), *extra,
    ]))
    folder = Path(result["report"]).parent
    return load_json(folder / "eligibility.json"), folder


def snapshot(case, records, *, issues=None, manifest=None):
    original = load_json(case["prepared"]) if manifest is None else manifest
    destination = case["root"] / f"variant-{len(list(case['root'].glob('variant-*')))}"
    return write_snapshot(destination, records, original["sources"], original["lineage"], issues or [], view=original["view"])


def row(report, label="isolador", dataset="fixture"):
    return next(r for r in report["by_class"] if r["class_id"] == label and r["dataset_id"] == dataset)


def test_report_counts_repeats_and_preserves_inputs(prepared):
    before = {str(p): file_hash(p) for root in (prepared["raw"], prepared["prepared"].parent)
              for p in root.rglob("*") if p.is_file()}
    report, folder = run_report(prepared)
    assert row(report)["eligible_images"] == 3
    assert row(report)["valid_bbox_annotations"] == 3
    assert row(report)["attributes"] == {"material": {"unknown": 3}}
    assert row(report, "cabo")["class_presence"] == "no_mapped_occurrences"
    assert row(report, "cabo")["eligible_negative_images"] == 0
    assert report["classes_without_mapped_occurrences"] == ["cabo", "torre"]
    assert report["prepared_view"]["eligible_now"]
    assert not report["reference_dataset"]["images_available"]
    repeated, second = run_report(prepared)
    assert report == repeated and folder != second
    for name in ("eligibility.md", "eligibility.json", "eligibility_by_class.csv", "eligibility_issues.jsonl"):
        assert (folder / name).read_bytes() == (second / name).read_bytes()
    assert all(file_hash(Path(p)) == sha for p, sha in before.items())
    assert load_json(folder / "status.json")["state"] == "completed"
    assert load_json(folder / "config.json")["taxonomy"] == str(prepared["root"] / "taxonomy.yaml")
    with pytest.raises(FileExistsError):
        write_reports(folder, report, [])


def test_negatives_require_exhaustive_coverage_and_view_union(prepared):
    _, records = read_snapshot(prepared["prepared"])
    records[0].annotations.clear()
    records[1].annotation_scope["isolador"] = "partial"
    records[2].annotation_scope.clear()
    path = snapshot(prepared, records)
    report, _ = run_report(prepared, path)
    assert row(report)["eligible_negative_images"] == 1
    assert row(report)["eligible_positive_images"] == 0
    assert row(report)["coverage"] == {"exhaustive": 1, "partial": 1, "unknown": 1}
    assert row(report)["blocked_images"] == 2
    assert not report["prepared_view"]["eligible_now"]


def test_other_labels_ignore_regions_geometry_and_deduplicated_counts(prepared):
    _, records = read_snapshot(prepared["prepared"])
    pending = deepcopy(records[0].annotations[0])
    pending.annotation_id, pending.original_label, pending.label = "pending", "unknown", None
    pending.mapping_status = "pending"
    records[0].annotations.append(pending)
    records[0].annotations[0].geometry.coordinates = [-1, 2, 8, 20]
    records[0].problems.extend(["split_missing", "split_missing"])
    ignored = deepcopy(records[1].annotations[0])
    ignored.annotation_id, ignored.original_label = "void", "void"
    ignored.mapping_status, ignored.label, ignored.flags = "excluded", None, {"ignore": True}
    records[1].annotations.append(ignored)
    records[2].annotations[0].geometry.type = "polygon"
    records[2].annotations[0].geometry.coordinates = [[0, 0], [10, 0], [0, 10]]
    records[2].annotations[0].geometry.convention = "pixel_xy"
    report, folder = run_report(prepared, snapshot(prepared, records))
    assert row(report)["eligible_images"] == 0
    assert row(report)["blocking_images_by_code"]["split_missing"] == 1
    assert row(report)["geometries"] == {"bbox": 2, "polygon": 1}
    findings = [json.loads(line) for line in (folder / "eligibility_issues.jsonl").read_text().splitlines()]
    assert any(f.get("annotation_id") == "pending" and "class:isolador" in f["contexts"] for f in findings)
    assert any(f.get("annotation_id") == "void" and f["code"] == "unsupported_ignore_region" for f in findings)
    assert {r["mapping_status"] for r in report["original_labels"]} == {"mapped", "pending", "excluded"}


def test_group_leakage_marks_both_sources_and_duplicate_ids(prepared):
    manifest, records = read_snapshot(prepared["prepared"])
    other = deepcopy(records[0])
    other.image_id, other.dataset_id = "other-image", "other"
    records[0].split, other.split = "train", "test"
    records.append(other)
    manifest["sources"]["other"] = deepcopy(manifest["sources"]["fixture"])
    report, folder = run_report(prepared, snapshot(prepared, records, manifest=manifest))
    assert row(report)["blocking_images_by_code"]["group_crosses_splits"] == 1
    assert row(report, dataset="other")["blocking_images_by_code"]["group_crosses_splits"] == 1
    assert row(report)["blocking_images_by_code"]["duplicate_annotation_id"] == 1
    findings = [json.loads(line) for line in (folder / "eligibility_issues.jsonl").read_text().splitlines()]
    assert {f["image_id"] for f in findings if f["code"] == "group_crosses_splits"} == {records[0].image_id, other.image_id}
    assert not report["prepared_view"]["eligible_now"]


@pytest.mark.parametrize("known_source", [True, False])
def test_source_issues_without_images_are_not_discarded(prepared, known_source):
    _, records = read_snapshot(prepared["prepared"])
    source = str(prepared["raw"] / "bad.json") if known_source else "/obsolete/unknown.json"
    issue = {"codes": ["unresolved_source_issue"], "source_issue": {"code": "annotation_without_image", "source": source}}
    report, folder = run_report(prepared, snapshot(prepared, records, issues=[issue]))
    assert row(report)["eligible_images"] == 0
    assert report["global_blockers"] == ([] if known_source else ["unresolved_source_issue"])
    assert not report["prepared_view"]["eligible_now"]
    assert source in (folder / "eligibility_issues.jsonl").read_text()


def test_coarse_mapping_and_curation_exclusions_are_separate(prepared):
    _, records = read_snapshot(prepared["prepared"])
    records[0].metadata["excluded"] = True
    records[1].annotations[0].flags["excluded_by_curation"] = True
    records[1].problems.append("annotation_exclusion_requires_image_review")
    tower = deepcopy(records[2].annotations[0])
    tower.annotation_id, tower.label, tower.original_label = "tower", "torre", "generic_tower"
    tower.attributes, tower.flags = {}, {"mapping_level": "coarse"}
    records[2].annotations.append(tower)
    report, _ = run_report(prepared, snapshot(prepared, records))
    assert report["totals"]["images"] == 3 and report["totals"]["excluded_images"] == 1
    assert report["datasets"]["fixture"]["excluded_image_annotations"] == 1
    assert row(report)["annotations_excluded_by_curation"] == 1
    assert row(report)["eligible_negative_images"] == 0
    assert row(report, "torre")["mapping_level"] == "coarse"
    assert row(report, "torre")["attributes"] == {}


def test_tiles_do_not_multiply_original_counts(project):
    config = project["config"]
    config["tiling"] = {"enabled": True, "size": 20, "overlap": .25, "border_policy": "clip"}
    save_yaml(project["prepare"], config)
    result = execute(parser().parse_args(["prepare", "--config", str(project["prepare"])]))
    cfg = save_yaml(project["root"] / "eligibility.yaml", {"taxonomy": "taxonomy.yaml", "mapping": "mapping.yaml"})
    case = {**project, "prepared": Path(result["manifest"]), "report_config": cfg}
    report, _ = run_report(case, extra=("--runs-root", str(project["root"] / "diagnostics")))
    assert report["totals"]["images"] == 3 and report["totals"]["annotations"] == 3
    assert report["totals"]["tiles_not_counted"] > 3
    assert row(report)["eligible_images"] == 3
    assert list(report["datasets"]) == ["fixture"]


@pytest.mark.parametrize("target,mode", [("image", "missing"), ("image", "changed"), ("annotation", "missing"), ("annotation", "changed")])
def test_source_integrity_blocks_report_without_aborting(prepared, target, mode):
    _, records = read_snapshot(prepared["prepared"])
    path = prepared["raw"] / (records[0].path if target == "image" else "annotations.json")
    if mode == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed")
    report, folder = run_report(prepared)
    assert row(report)["eligible_images"] == (2 if target == "image" else 0)
    assert not report["prepared_view"]["eligible_now"]
    assert load_json(folder / "status.json")["state"] == "completed"


@pytest.mark.parametrize("target", ["taxonomy.yaml", "mapping.yaml", "images.jsonl", "manifest.json"])
def test_invalid_snapshot_or_configuration_records_failure(prepared, target):
    if target.endswith("yaml"):
        path = prepared["root"] / target
        content = read_config(path)
        content["version"] = "different"
        save_yaml(path, content)
    else:
        path = prepared["prepared"].parent / target
        if target == "images.jsonl":
            path.write_text(path.read_text() + "\n")
        else:
            path.write_text("invalid")
    assert main(["eligibility", "--config", str(prepared["report_config"]), "--manifest", str(prepared["prepared"])]) == 1
    statuses = [load_json(p) for p in (prepared["root"] / "reports").glob("*/status.json")]
    assert len(statuses) == 1 and statuses[0]["state"] == "failed"
    assert not list((prepared["root"] / "reports").glob("*/eligibility.json"))


def test_unprepared_snapshot_and_output_in_source_rejected(prepared):
    with pytest.raises(ValueError, match="produzido por prepare"):
        run_report(prepared, prepared["manifest"])
    with pytest.raises(ValueError, match="protegida"):
        run_report(prepared, extra=("--runs-root", str(prepared["raw"] / "reports")))


def test_order_independence_and_no_documentation_reads(prepared, monkeypatch):
    manifest, records = read_snapshot(prepared["prepared"])
    original = Path.open
    def checked_open(path, *args, **kwargs):
        assert "doc" not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked_open)
    first = build_report(manifest, records, prepared["taxonomy"], [])
    for r in records:
        r.annotations.reverse()
    second = build_report(manifest, list(reversed(records)), prepared["taxonomy"], [])
    assert first == second
    report, _ = run_report(prepared)
    assert report["prepared_view"]["eligible_now"]


def test_malformed_bbox_is_a_diagnostic_not_an_exception(prepared):
    _, records = read_snapshot(prepared["prepared"])
    annotation = records[0].annotations[0]
    annotation.geometry.coordinates = None
    assert "invalid_geometry" in bbox_errors(annotation, records[0])
    assert "invalid_geometry" in eligibility(records, ["isolador"])[0]["codes"]


def test_class_eligibility_does_not_approve_combined_view(prepared):
    manifest, records = read_snapshot(prepared["prepared"])
    records[1].annotations[0].label = "cabo"
    records[1].annotation_scope = {"cabo": "exhaustive"}
    records[2].annotation_scope = {}
    manifest["view"]["classes"] = ["cabo", "isolador"]
    report, _ = run_report(prepared, snapshot(prepared, records, manifest=manifest))
    assert row(report)["eligible_images"] == 1
    assert row(report, "cabo")["eligible_images"] == 1
    assert report["prepared_view"]["blocked_images"] == 3
    assert not report["prepared_view"]["eligible_now"]


def test_baseline_difference_is_reported_without_forcing_counts(prepared):
    baseline = prepared["root"] / "baseline.csv"
    baseline.write_text("dataset,images,annotations\nfixture,99,99\n")
    config = read_config(prepared["report_config"])
    config["baseline"] = "baseline.csv"
    save_yaml(prepared["report_config"], config)
    report, folder = run_report(prepared)
    assert not report["reconciliation"]["matches"]
    assert report["totals"]["images"] == 3
    assert "há diferenças" in (folder / "eligibility.md").read_text()


def test_empty_view_and_persisted_group_conflict(prepared):
    manifest, records = read_snapshot(prepared["prepared"])
    records[0].group_id = records[1].group_id = "conflicting"
    records[0].split = records[1].split = None
    problem = {"code": "group_crosses_splits", "group_id": "conflicting", "splits": ["train", "test"]}
    report, _ = run_report(prepared, snapshot(prepared, records, issues=[problem]))
    assert row(report)["blocking_images_by_code"]["group_crosses_splits"] == 2
    manifest["view"]["image_ids"] = []
    manifest["view"]["evaluation_image_ids"] = []
    report, _ = run_report(prepared, snapshot(prepared, records, manifest=manifest, issues=[{"codes": ["empty_view"]}]))
    assert "empty_view" in report["prepared_view"]["blocking_codes"]
    assert not report["prepared_view"]["eligible_now"]
