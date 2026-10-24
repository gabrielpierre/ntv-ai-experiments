import copy
import fcntl
import os
import shutil
import subprocess
import contextlib
import json
from functools import lru_cache
from dataclasses import asdict
from pathlib import Path

from ntv_ai.domain.records import ArtifactReference, PreparedDataset
from ntv_ai.experiments.recipes import checked_reference, compatibility, local_reference, resolve_recipe, validate_recipe
from ntv_ai.experiments.runs import Run
from ntv_ai.models.common import artifact, prepared_dataset, verify_framework_images
from ntv_ai.storage import atomic_json, file_hash, load_json, read_snapshot, safe_output


@lru_cache(maxsize=None)
def environment_python(environment):
    conda = shutil.which("conda") or "/opt/miniconda3/bin/conda"
    result = subprocess.run([conda, "run", "-n", environment, "python", "-c",
                             "import sys; print(sys.executable)"], capture_output=True, text=True, check=True)
    path = result.stdout.strip().splitlines()[-1]
    if not Path(path).is_file():
        raise RuntimeError(f"Missing interpreter for {environment}")
    return path


def preflight(config):
    config = copy.deepcopy(config)
    validate_recipe(config)
    root = Path(config["runtime"]["storage_root"])
    if config["dataset"].get("fixture"):
        if config["dataset"]["fixture"] != "coco8":
            raise ValueError("Fixture desconhecido")
        descriptor = root / "data/fixtures/coco8/dataset.json"
        if not descriptor.is_file():
            raise ValueError("COCO8 não preparado; execute acquire explicitamente")
        config["dataset"] = {**load_json(descriptor), "fixture": "coco8"}
    data = config["dataset"]
    data["manifest"]["path"] = checked_reference(data["manifest"], root)
    manifest, _ = read_snapshot(Path(data["manifest"]["path"]))
    if manifest["version"] != data["version"]:
        raise ValueError("Versão do dataset divergente")
    from ntv_ai.annotator.lofo import preflight_guard
    preflight_guard(config, manifest, Path(data["manifest"]["path"]))
    data["source_roots"] = {k: str((root / v).resolve()) for k, v in data.get("source_roots", {}).items()}
    dataset = PreparedDataset(ArtifactReference(**data["manifest"]), data["view_id"], tuple(data["classes"]))
    manifest, records = prepared_dataset(dataset, data["source_roots"])
    if manifest['view'].get('purpose') == 'exploratory' and config['purpose'] != 'exploratory':
        raise ValueError('Dados exploratórios exigem finalidade exploratory')
    if not all(any(r.split == split for r in records) for split in ("train", "val")):
        raise ValueError("Treinamento requer train e val não vazios")
    verify_framework_images(records, manifest["sources"])
    model = config["model"]
    model["weights"]["path"] = local_reference(model["weights"], root)
    if "implementation" in model:
        implementation = model["implementation"]
        path = (root / implementation["path"]).resolve()
        if not path.is_dir():
            raise ValueError(f"Implementação externa ausente: {path}")
        implementation["path"] = str(path)
    return config


def resume_configuration(config, parent):
    parent = Path(parent).resolve()
    previous = load_json(parent / "config.json")
    status = load_json(parent / "status.json")
    result_path = parent / "result.json"
    if not result_path.exists():
        result_path = parent / "progress.json"
    result = load_json(result_path)
    # Changing epochs is intentional continuation; other recipe changes create a new experiment.
    old, new = copy.deepcopy(previous), copy.deepcopy(config)
    old["training"]["epochs"] = new["training"]["epochs"]
    if compatibility(old) != compatibility(new) or old["training"]["seed"] != new["training"]["seed"]:
        raise ValueError("Retomada incompatível com dados, protocolo, modelo, ambiente ou receita")
    if config["training"]["epochs"] <= result["last_epoch"] + 1:
        raise ValueError("A retomada deve incluir épocas ainda não executadas")
    ref = result["resume"]
    local_reference(ref, Path("/"))
    config["resume"] = {"parent_run": status["run_id"], "checkpoint": ref,
                        "selection": str(parent / "selection"), "last_epoch": result["last_epoch"]}
    if config['training'].get('control'):
        control = result.get('control')
        if not control:
            raise ValueError('Retomada requer o estado do controle comum')
        local_reference(control, Path('/'))
        state = load_json(Path(control['path']))
        if state['config'] != config['training']['control'] or state['state']['epoch'] != result['last_epoch']:
            raise ValueError('Estado de controle incompatível com checkpoint')
        if state['state']['stopped']:
            raise ValueError('Treino concluído por early stopping; não reiniciar a paciência em retomada')
        config['resume']['control'] = control
    return config


def launch(operation, config, runs_root=None, checkpoint=None, resume=None, split="val", target=None, source_run=None,
           on_created=None, execution=None):
    # Validation runs in the light environment, before importing or spawning a framework.
    config = preflight(config)
    if execution and execution.get("cpu_threads"):
        config["runtime"]["cpu_threads"] = execution["cpu_threads"]
    if resume:
        config = resume_configuration(config, resume)
    root = Path(config["runtime"]["storage_root"])
    runs_root = Path(runs_root or root / "runs").resolve()
    sources = read_snapshot(Path(config["dataset"]["manifest"]["path"]))[0]["sources"]
    protected = [Path(s["root"]) for s in sources.values()]
    protected += [Path(v) for v in config["dataset"].get("source_roots", {}).values()]
    if target:
        from ntv_ai.experiments.cross import load_dataset
        target_manifest, _ = load_dataset(target)
        protected += [Path(s['root']) for s in target_manifest['sources'].values()]
        protected.append(Path(target['manifest']['path']).parent)
    safe_output(runs_root, [*protected, Path(config["dataset"]["manifest"]["path"]).parent])
    if operation in ("predict", "measure", "cross_predict"):
        if checkpoint is None:
            raise ValueError("Informe --checkpoint de uma execução treinada")
        checkpoint = Path(checkpoint).resolve()
        checkpoint_ref = asdict(artifact(checkpoint))
        config["inference_checkpoint"] = checkpoint_ref
    else:
        checkpoint_ref = None
    if operation == "profile":
        if config["training"]["batch"] != 1 or not config["training"]["amp"] or config["training"]["epochs"] != 1:
            raise ValueError("Perfil curto requer batch 1, AMP e uma época; crie uma receita explícita")
    with Run(runs_root, operation, config) as run:
        print(f"run_id: {run.id} ({config['model']['backend']}, seed {config['training']['seed']})", flush=True)
        request = {"operation": operation, "config": config, "output": str(run.path), "split": split,
                   "checkpoint": str(checkpoint) if checkpoint else None}
        if execution:
            request["lease_fd"] = execution.get("lease_fd")
            atomic_json(run.path / "execution.json", {k: v for k, v in execution.items() if k != "lease_fd"})
        if operation == 'cross_predict':
            from ntv_ai.experiments.cross import pair_ready
            ready = pair_ready(config['dataset'], target, split)
            if ready['state'] != 'pending':
                raise ValueError(str(ready))
            request.update(target=target, source_run=source_run)
            atomic_json(run.path / 'target.json', target)
        atomic_json(run.path / "request.json", request)
        if on_created:
            on_created(run.path)
        runtime = config["runtime"]
        if execution and execution.get("lease_fd") is not None:
            command = [runtime.get("python") or environment_python(runtime["environment"])]
        elif runtime.get("python"):
            command = [runtime["python"]]
        else:
            conda = shutil.which("conda")
            if conda is None:
                raise RuntimeError("Conda não encontrado; configure runtime.python")
            command = [conda, "run", "--no-capture-output", "-n", runtime["environment"], "python"]
        command += ["-m", "ntv_ai.pipelines.experiment_worker", str(run.path / "request.json")]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONHASHSEED"] = str(config["training"]["seed"])
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if runtime.get("cpu_threads"):
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[name] = str(runtime["cpu_threads"])
        lease = execution.get("lease_fd") if execution else None
        lock_context = contextlib.nullcontext() if lease is not None else (runs_root / f".gpu-{runtime['device']}.lock").open("a")
        with lock_context as lock:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            with (run.path / "worker.log").open("w") as log:
                if execution is None:
                    child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                else:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                             pass_fds=(lease,) if lease is not None else ())
                    process = {"pid": child.pid, "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                               "start_ticks": Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]}
                    atomic_json(run.path / "worker-process.json", process)
                    child.wait()
        run.status["artifacts"].update(log=str(run.path / "worker.log"), config=str(run.path / "config.json"))
        if child.returncode:
            failure_path = run.path / "failure.json"
            failure = load_json(failure_path) if failure_path.exists() else {"message": f"Worker terminou com código {child.returncode}", "oom": False}
            run.status["failure"] = failure
            raise RuntimeError(f"{run.id}: {failure['message']}; consulte {run.path / 'worker.log'}")
        result = load_json(run.path / "result.json")
        result.update(run_id=run.id, run_path=str(run.path), compatibility=compatibility(config))
        atomic_json(run.path / "result.json", result, replace=True)
        run.status["artifacts"]["result"] = str(run.path / "result.json")
        return result


def run_experiment(operation, config_path, **kwargs):
    storage_root = kwargs.pop("storage_root", None)
    return launch(operation, resolve_recipe(Path(config_path), storage_root), **kwargs)
