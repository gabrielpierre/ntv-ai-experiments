from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from ntv_ai.cli import main, parser
from ntv_ai.domain import Detection
from ntv_ai.experiments.control import DEFAULT_CONTROL, TrainingControl
from ntv_ai.experiments.recipes import compatibility, resolve_recipe
from ntv_ai.experiments.selection import CheckpointSelection
from ntv_ai.models.inference import predict_image
from ntv_ai.models.package import ModelPackage
from ntv_ai.pipelines import model_packages
from ntv_ai.pipelines.experiments import preflight, resume_configuration
from ntv_ai.storage import file_hash, load_json, write_snapshot
from test_experiment_core import prepared, recipe


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def update_metadata(root, relative, transform):
    index = load_json(root / "retomada/indice.json")
    ref = next(r for r in index["runs"][0]["files"] if r["run_relative_path"] == relative)
    path = root / ref["package_path"]
    value = load_json(path)
    transform(value)
    save(path, value)


@pytest.fixture
def checkpoint_package(tmp_path):
    old = tmp_path / "original"
    run = old / "runs/old-run"
    root = tmp_path / "package with spaces"
    native = root / "retomada/dfine-s/full640/seed-42"
    native.mkdir(parents=True)
    config = recipe()
    config.update(purpose="exploratory", provenance={"layers": {}, "recipe_sha256": "a" * 64})
    config["runtime"].update(storage_root=str(old), environment="ntv-dfine")
    config["protocol"]["resolution"] = 640
    config["training"].update(epochs=10, control=DEFAULT_CONTROL)
    config["model"]["weights"]["path"] = str(old / "data/initial.pth")
    config["dataset"] = {"classes": ["isolador"], "view_id": "test", "version": "v1",
                         "manifest": {"path": str(old / "data/snapshot/manifest.json"), "sha256": "b" * 64}}
    full = native / "backend/resume-0000.pth"
    full.parent.mkdir(); full.write_bytes(b"fake optimizer model rng")
    best = root / "dfine-s/full640/seed-42.pth"
    best.parent.mkdir(parents=True); best.write_bytes(b"fake ema")
    state = TrainingControl(DEFAULT_CONTROL)
    state.observe(0, .3)
    control = full.with_suffix(".pth.control.json")
    save(control, {"resume_sha256": file_hash(full), "config": DEFAULT_CONTROL, "state": state.state})
    selected = {"epoch": 0, "mAP50_95": .3, "checkpoint": {"path": str(run / "selection/best.pth"), "sha256": file_hash(best)}}
    save(native / "config.json", config)
    save(native / "status.json", {"run_id": "old-run", "state": "interrupted", "artifacts": {}})
    save(native / "environment.json", {"code_sha": "old-code"})
    save(native / "result.json", {"best": selected, "last_epoch": 0,
        "resume": {"path": str(run / "backend" / full.name), "sha256": file_hash(full)},
        "control": {"path": str(run / "backend" / control.name), "sha256": file_hash(control)}})
    save(native / "selection/selection.json", {"best": selected, "history": [{"epoch": 0, "mAP50_95": .3}]})
    refs = []
    for path in sorted(native.rglob("*")):
        if not path.is_file(): continue
        role = "resume" if path == full else "control" if path == control else "metadata"
        refs.append({"role": role, "package_path": path.relative_to(root).as_posix(),
                     "run_relative_path": path.relative_to(native).as_posix()})
    refs.append({"role": "best_inference", "package_path": best.relative_to(root).as_posix(),
                 "run_relative_path": "selection/best.pth"})
    save(root / "retomada/indice.json", {"schema_version": 1, "source_root": str(old), "runs": [{
        "id": "ntv_ai/full640/dfine-s/42", "run_id": "old-run", "backend": "dfine", "protocol": "full640",
        "seed": 42, "control_stopped": False, "original_run_path": str(run), "files": refs}]})
    return root


def test_list_without_checksum_files_and_framework_free_cli(checkpoint_package):
    p = ModelPackage(checkpoint_package)
    assert p.list_models()["models"][0]["model"] == "dfine-s"
    assert not (checkpoint_package / "SHA256SUMS").exists()
    code = "from ntv_ai.cli import main; import sys; assert main(sys.argv[1:]) == 0; assert not {'torch','rfdetr','ultralytics'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code, "models", "list", "--package", str(checkpoint_package)], check=True, capture_output=True)
    with pytest.raises(SystemExit):
        parser().parse_args(["models", "verify", "--package", str(checkpoint_package)])


def test_changed_checkpoints_and_stale_checksums_do_not_block_restore_or_resume(checkpoint_package, tmp_path):
    (checkpoint_package / "SHA256SUMS").write_text("obsolete, unused manifest")
    p = ModelPackage(checkpoint_package)
    row = p.select("dfine-s", "full640", 42)
    p.path(p.reference(row, "resume")["package_path"]).write_bytes(b"changed")
    p.path(p.reference(row, "best_inference")["package_path"]).write_bytes(b"new ema")
    output = tmp_path / "restored"
    p.restore(row, output, tmp_path / "new-root")
    assert (output / "backend/resume-0000.pth").read_bytes() == b"changed"
    resumed = resume_configuration(resolve_recipe(output / "recipe.json"), output)
    assert resumed["resume"]["parent_run"] == "old-run"
    selection = CheckpointSelection(tmp_path / "selection", output / "selection")
    assert Path(selection.best["checkpoint"]["path"]).read_bytes() == b"new ema"


def test_missing_checkpoint_fails_before_output(checkpoint_package, tmp_path):
    p = ModelPackage(checkpoint_package)
    row = p.select("dfine-s", "full640", 42)
    p.path(p.reference(row, "resume")["package_path"]).unlink()
    with pytest.raises(ValueError, match="Arquivo ausente"):
        p.restore(row, tmp_path / "restored", tmp_path)
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "a/../escape", "a//escape", "a\\escape"])
def test_paths_cannot_escape_package(checkpoint_package, name):
    index = load_json(checkpoint_package / "retomada/indice.json")
    index["runs"][0]["files"][0]["package_path"] = name
    save(checkpoint_package / "retomada/indice.json", index)
    with pytest.raises(ValueError, match="Caminho relativo"):
        ModelPackage(checkpoint_package)


def test_symlink_and_duplicate_selection_rejected(checkpoint_package, tmp_path):
    p = checkpoint_package / "dfine-s/full640/seed-42.pth"
    target = tmp_path / "outside"; shutil.move(p, target); p.symlink_to(target)
    with pytest.raises(ValueError, match="simbólico"):
        package = ModelPackage(checkpoint_package)
        package.restore(package.select("dfine-s", "full640", 42), tmp_path / "restored", tmp_path)
    p.unlink(); shutil.move(target, p)
    idx = load_json(checkpoint_package / "retomada/indice.json")
    extra = deepcopy(idx["runs"][0]); extra["id"] = "duplicate"
    idx["runs"].append(extra); save(checkpoint_package / "retomada/indice.json", idx)
    with pytest.raises(ValueError, match="ambígua"):
        ModelPackage(checkpoint_package).select("dfine-s", "full640", 42)


def test_restore_rebases_paths_and_generates_cli_recipe(checkpoint_package, tmp_path):
    package = ModelPackage(checkpoint_package); row = package.select("dfine-s", "full640", 42)
    before = {p.relative_to(checkpoint_package): p.read_bytes() for p in checkpoint_package.rglob("*") if p.is_file()}
    output, new_root = tmp_path / "restored", tmp_path / "new-root"
    report = package.restore(row, output, new_root)
    config = resolve_recipe(output / "recipe.json")
    assert "provenance" not in load_json(output / "recipe.json")
    assert config["dataset"]["manifest"]["path"] == str(new_root / "data/snapshot/manifest.json")
    assert compatibility(config) == compatibility(package.config(row))
    assert report["missing_artifacts"] and not report["training_stopped"]
    resumed = resume_configuration(config, output)
    assert resumed["resume"]["checkpoint"]["path"] == str(output / "backend/resume-0000.pth")
    assert resumed["resume"]["parent_run"] == "old-run"
    assert resumed["resume"]["control"]["path"] == str(output / "backend/resume-0000.pth.control.json")
    selection = CheckpointSelection(tmp_path / "next-selection", output / "selection")
    assert selection.best["mAP50_95"] == .3
    assert {p.relative_to(checkpoint_package): p.read_bytes() for p in checkpoint_package.rglob("*") if p.is_file()} == before
    with pytest.raises(FileExistsError): package.restore(row, output, new_root)
    with pytest.raises(ValueError, match="protegida"): package.restore(row, checkpoint_package / "output", new_root)


def test_restore_preserves_early_stopping(checkpoint_package, tmp_path):
    update_metadata(checkpoint_package, "backend/resume-0000.pth.control.json", lambda c: c["state"].update(stopped=True))
    idx = load_json(checkpoint_package / "retomada/indice.json")
    row = idx["runs"][0]; row["control_stopped"] = True
    save(checkpoint_package / "retomada/indice.json", idx)
    p = ModelPackage(checkpoint_package); row = p.select("dfine-s", "full640", 42)
    report = p.restore(row, tmp_path / "stopped", tmp_path / "new-root")
    assert report["training_stopped"]
    with pytest.raises(ValueError, match="early stopping"):
        resume_configuration(resolve_recipe(Path(report["recipe"])), Path(report["restored_run"]))


def test_restore_sources_without_rewriting_manifest(checkpoint_package, tmp_path):
    old = Path(load_json(checkpoint_package / "retomada/indice.json")["source_root"])
    new = tmp_path / "new-root"
    manifest = write_snapshot(new / "data/snapshot", [], {"source": {"root": str(old / "data/raw")}}, {}, [], view={"eligible": True})
    checksum = file_hash(manifest)
    package = ModelPackage(checkpoint_package)
    report = package.restore(package.select("dfine-s", "full640", 42), tmp_path / "restored", new)
    restored = load_json(Path(report["recipe"]))
    assert restored["dataset"]["source_roots"] == {"source": str(new / "data/raw")}
    assert file_hash(manifest) == checksum


def test_image_tiles_reproject_merge_and_keep_negatives():
    calls = []
    def predictor(image):
        left = [0, 6, 8][len(calls)]; calls.append(image.size)
        values = [Detection("a", [8-left, 2, 10-left, 4], .9)]
        if left == 8: values.append(Detection("b", [0, 2, 2, 4], .8))
        return values, .01
    items, timing = predict_image(predictor, Image.new("RGB", (20, 12)), {"tiling": True, "size": 12, "overlap": .5}, .5)
    assert items == [Detection("a", [8, 2, 10, 4], .9), Detection("b", [8, 2, 10, 4], .8)]
    assert len(calls) == timing["samples"] == 3 and timing["model_seconds"] == pytest.approx(.03)
    items, timing = predict_image(lambda im: ([], 0), Image.new("RGB", (20, 12)), {"tiling": True, "size": 12, "overlap": .5}, .5)
    assert not items and timing["samples"] == 3
    items, timing = predict_image(lambda im: ([], 0), Image.new("RGB", (20, 12)), {"tiling": False}, .5)
    assert not items and timing["samples"] == 1


def test_predict_without_training_assets_and_records_failure(checkpoint_package, tmp_path, monkeypatch):
    source = tmp_path / "image.png"; Image.new("RGB", (30, 20)).save(source)
    original = model_packages.subprocess.run
    requests = []
    def worker(command, **kwargs):
        if "ntv_ai.pipelines.package_worker" not in command: return original(command, **kwargs)
        request = load_json(Path(command[-1])); requests.append(request)
        save(Path(request["output"]) / "failure.json", {"message": "CUDA out of memory", "oom": True, "recipe_changed": False})
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(model_packages.subprocess, "run", worker)
    with pytest.raises(RuntimeError, match="out of memory"):
        model_packages.predict_package(checkpoint_package, "dfine-s", "full640", 42, source,
            storage_root=tmp_path / "new-root", runs_root=tmp_path / "runs", python=sys.executable)
    assert len(requests) == 1
    assert "weights" not in requests[0]["config"]["model"]
    assert requests[0]["config"]["protocol"]["resolution"] == 640
    assert load_json(Path(requests[0]["output"]) / "status.json")["state"] == "failed"


def test_cli_selectors_and_invalid_index(checkpoint_package, capsys):
    args = parser().parse_args(["models", "predict", "--package", str(checkpoint_package), "--model", "dfine-s",
                               "--protocol", "full640", "--seed", "42", "--source", "image.png"])
    assert args.seed == 42 and args.device == 0
    with pytest.raises(ValueError, match="não encontrado"):
        ModelPackage(checkpoint_package).select("dfine-s", "full640", 99)
    (checkpoint_package / "retomada/indice.json").write_text("{}")
    assert main(["models", "list", "--package", str(checkpoint_package)]) == 1
    assert "Índice de modelos não suportado" in capsys.readouterr().err


def test_package_evaluation_uses_original_test_and_keeps_head(checkpoint_package, project, tmp_path, monkeypatch):
    from test_cross_benchmark import pair
    _, target = pair(project, tmp_path)
    original = model_packages.subprocess.run
    calls = []
    def worker(command, **kwargs):
        if 'ntv_ai.pipelines.package_worker' not in command:
            return original(command, **kwargs)
        request = load_json(Path(command[-1]))
        calls.append(request)
        assert request['config']['classes'] == ['isolador']
        assert request['evaluation_classes'] == ['isolador']
        assert request['split'] == 'test'
        assert len(request['images']) == 2  # Includes the negative original.
        save(Path(request['output']) / 'result.json', {'predictions': {'path': 'synthetic'}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(model_packages.subprocess, 'run', worker)
    model_packages.predict_package(checkpoint_package, 'dfine-s', 'full640', 42, None,
        evaluation_target=target, storage_root=tmp_path / 'new-root', runs_root=tmp_path / 'runs', python=sys.executable)
    assert len(calls) == 1


def test_training_accepts_local_weights_and_edited_external_code(project, tmp_path):
    dataset, manifest, _ = prepared(project)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"local weights")
    implementation = tmp_path / "dfine"
    implementation.mkdir()
    (implementation / "model.py").write_text("# local modifications, no git checkout\n")
    config = recipe()
    config["dataset"] = {"manifest": asdict(dataset.manifest), "view_id": dataset.view_id,
                         "version": manifest["version"], "classes": list(dataset.classes)}
    config["model"].update(weights={"path": str(weights)},
                           implementation={"path": str(implementation), "revision": "obsolete"},
                           package={"name": "example", "version": "1", "wheel": {"path": "missing.whl"}})
    resolved = preflight(config)
    assert resolved["model"]["weights"]["path"] == str(weights)
    model_packages.check_external_implementation(resolved)
    weights.unlink()
    with pytest.raises(ValueError, match="Arquivo ausente"):
        preflight(config)


def test_runtime_checks_versions_without_comparing_installed_files(monkeypatch):
    import importlib.metadata
    from ntv_ai.models.runtime import verify_implementation
    versions = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: SimpleNamespace(version="1"))
    config = recipe()
    config["model"]["package"] = {"name": "example", "version": "1", "wheel": {"path": "missing.whl"}}
    verify_implementation(config)
