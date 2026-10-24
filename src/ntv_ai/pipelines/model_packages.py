"""Launch inference from a downloaded package in the selected backend environment."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess

from PIL import Image

from ntv_ai.experiments.runs import Run
from ntv_ai.models.package import ModelPackage, PROTOCOLS
from ntv_ai.storage import atomic_json, load_json, safe_output


ENVIRONMENTS = {"yolo26": "ntv-yolo26l", "dfine": "ntv-dfine", "rfdetr": "ntv-rfdetr"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def source_images(source):
    source = Path(source).resolve()
    paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()) if source.is_dir() else [source]
    if not paths:
        raise ValueError("Nenhuma imagem encontrada")
    rows = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        name = path.relative_to(source).as_posix() if source.is_dir() else path.name
        rows.append({"image_id": name, "name": name, "path": str(path), "width": width, "height": height})
    return rows


def prediction_config(package, row, storage_root, device):
    if type(device) is not int or device < 0:
        raise ValueError("Informe um único índice de GPU")
    config = deepcopy(package.config(row))
    config["classes"] = config["dataset"]["classes"]
    config["runtime"] = {"storage_root": str(Path(storage_root).resolve()), "device": device,
                         "environment": ENVIRONMENTS[row["backend"]]}
    # Inference uses the trained weights, never the initial COCO weights or a dataset.
    config["model"].pop("weights", None)
    if "package" in config["model"]:
        config["model"]["package"].pop("wheel", None)
    implementation = config["model"].get("implementation")
    if implementation:
        source = Path(implementation["path"])
        old_root = Path(package.index["source_root"])
        if source.is_absolute():
            if not source.is_relative_to(old_root):
                raise ValueError("Implementação externa fora da raiz original")
            source = source.relative_to(old_root)
        implementation["path"] = str(Path(storage_root).resolve() / source)
    return config


def check_external_implementation(config):
    ref = config["model"].get("implementation")
    if not ref:
        return
    path = Path(ref["path"])
    if not path.is_dir():
        raise ValueError(f"Código D-FINE ausente: {path}")


def predict_package(package_path, model, protocol, seed, source, *, storage_root=None,
                    runs_root=None, device=0, python=None, evaluation_target=None, split="test"):
    package = ModelPackage(package_path)
    row = package.select(model, protocol, seed)
    ref = package.reference(row, "best_inference")
    checkpoint = package.file(ref["package_path"])
    config = prediction_config(package, row, storage_root or Path.cwd(), device)
    check_external_implementation(config)
    if evaluation_target:
        from ntv_ai.experiments.cross import evaluation_data
        target = load_json(Path(evaluation_target)) if isinstance(evaluation_target, (str, Path)) else evaluation_target
        classes = [x for x in config['classes'] if x in target['classes']]
        manifest, originals, _ = evaluation_data(target, classes, split)
        images = [{"image_id": r.image_id, "name": r.path,
                   "path": str(Path(manifest['sources'][r.dataset_id]['root']) / r.path),
                   "width": r.width, "height": r.height} for r in originals]
    else:
        images = source_images(source)
    root = Path(runs_root or Path(config["runtime"]["storage_root"]) / "runs").resolve()
    safe_output(root, [package.root, *([Path(source).resolve()] if source is not None else [])])
    if python:
        command = [str(Path(python).resolve())]
    else:
        conda = shutil.which("conda")
        if not conda:
            raise ValueError("Conda não encontrado; informe --python com o Python do ambiente do modelo")
        command = [conda, "run", "--no-capture-output", "-n", config["runtime"]["environment"], "python"]
    config["package_origin"] = {"id": row["id"], "source_run": row["run_id"],
                                "protocol": protocol}
    with Run(root, "package_predict", config) as run:
        request = {"package": str(package.root), "model": model, "protocol": protocol, "seed": seed,
                   "config": config, "checkpoint": str(checkpoint), "images": images,
                   "tiling": PROTOCOLS[protocol], "output": str(run.path)}
        if evaluation_target:
            request.update(evaluation_target=target, evaluation_classes=classes, split=split)
        atomic_json(run.path / "request.json", request)
        atomic_json(run.path / "images.json", images)
        command += ["-m", "ntv_ai.pipelines.package_worker", str(run.path / "request.json")]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        with (run.path / "worker.log").open("w") as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        run.status["artifacts"].update(log=str(run.path / "worker.log"), images=str(run.path / "images.json"))
        if child.returncode:
            failure = load_json(run.path / "failure.json") if (run.path / "failure.json").exists() else {"message": f"Worker encerrou com código {child.returncode}"}
            run.status["failure"] = failure
            raise RuntimeError(f"{failure['message']}; consulte {run.path / 'worker.log'}")
        result = load_json(run.path / "result.json")
        run.status["artifacts"].update(result=str(run.path / "result.json"), predictions=result["predictions"]["path"])
        return {"run_id": run.id, "run_path": str(run.path), **result}
