import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import yaml

from ntv_ai.cli import execute, parser as core_parser
from ntv_ai.domain.records import PreparedDataset
from ntv_ai.experiments.config import read_config, reference
from ntv_ai.experiments.runs import Run
from ntv_ai.models.backends.yolo26l import YOLO26LBackend, artifact, prepared_dataset
from ntv_ai.storage import atomic_json, file_hash, inside, json_bytes, load_json, read_snapshot, safe_output


def save_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
    return path


def core(*args):
    return execute(core_parser().parse_args([str(a) for a in args]))


def collect_runtime(run, backend_output):
    runtime = backend_output / "runtime.json"
    if runtime.exists():
        environment = load_json(run.path / "environment.json")
        environment.update(load_json(runtime))
        atomic_json(run.path / "environment.json", environment, replace=True)


def run_model(operation, config_path, *, manifest_path=None, checkpoint_path=None, runs_root=None):
    config = read_config(config_path)
    manifest_path = (manifest_path or reference(config_path, config["manifest"])).resolve()
    checkpoint_path = (checkpoint_path or reference(config_path, config["checkpoint"])).resolve()
    root = runs_root or reference(config_path, config["runs"])
    raw_manifest = load_json(manifest_path)
    protected = [Path(s["root"]) for s in raw_manifest["sources"].values()]
    safe_output(root, [*protected, manifest_path.parent, checkpoint_path.parent])
    resolved = {**config, "manifest": asdict(artifact(manifest_path)),
                "checkpoint": asdict(artifact(checkpoint_path)), "recipe_sha256": file_hash(config_path)}
    with Run(root, f"yolo26l_{operation}", resolved) as run:
        print(f"run_id: {run.id} ({operation})", flush=True)
        manifest, _ = read_snapshot(manifest_path)
        view = manifest.get("view") or {}
        dataset = PreparedDataset(artifact(manifest_path), view.get("id", ""), tuple(view.get("classes", [])))
        backend = YOLO26LBackend()
        output = run.path / "backend"
        try:
            if operation == "train":
                result = backend.train(dataset, resolved, output)
                atomic_json(run.path / "result.json", asdict(result))
                run.status["artifacts"].update({k: ref.path for k, ref in result.checkpoints.items()})
                run.status["artifacts"]["training"] = str(output / "training.json")
                return {"run_id": run.id, **asdict(result)}
            manifest, selected = prepared_dataset(dataset)
            split = config.get("split", "val")
            if split not in ("train", "val", "test"):
                raise ValueError("Split de inferência inválido")
            predictions = backend.predict(artifact(checkpoint_path), [r for r in selected if r.split == split], {
                "classes": list(dataset.classes), "sources": manifest["sources"],
                "inference": config["inference"], "output": str(output),
            })
            prediction_path = run.path / "predictions.jsonl"
            with prediction_path.open("xb") as stream:
                for row in sorted(predictions, key=lambda row: row.image_id):
                    stream.write(json_bytes(asdict(row)))
            result = {"run_id": run.id, "predictions": asdict(artifact(prediction_path)),
                      "images": len(predictions), "detections": sum(len(p.detections) for p in predictions),
                      "dataset_version": manifest["version"], "split": split}
            atomic_json(run.path / "result.json", result)
            run.status["artifacts"]["predictions"] = str(prediction_path)
            return result
        finally:
            collect_runtime(run, output)


def download(spec, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with urllib.request.urlopen(spec["url"], timeout=60) as response, pending.open("xb") as stream:
        shutil.copyfileobj(response, stream)
        final_url = response.url
    sha256 = file_hash(pending)
    if spec.get("sha256") and sha256 != spec["sha256"]:
        raise ValueError(f"Hash do download divergente: {path.name}")
    os.link(pending, path)
    pending.unlink()
    return {"url": spec["url"], "resolved_url": final_url, **asdict(artifact(path))}


def acquire(config_path, config, root):
    with Run(root, "yolo26l_acquire", config["downloads"]) as run:
        output = reference(config_path, config["assets"]) / run.id
        output.mkdir(parents=True, exist_ok=False)
        evidence = {}
        for key, name in (("dataset", "coco8.zip"), ("weights", "yolo26l.pt"), ("metadata", "coco8.yaml")):
            evidence[key] = download(config["downloads"][key], output / name)
        atomic_json(output / "downloads.json", evidence)
        with zipfile.ZipFile(output / "coco8.zip") as archive:
            for member in archive.infolist():
                inside(output, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Link inesperado no arquivo COCO8")
            archive.extractall(output)
        native = read_config(output / "coco8.yaml")
        names = native["names"]
        classes = [names[i] for i in range(80)]
        if len(names) != 80 or len(set(classes)) != 80:
            raise ValueError("COCO8 deve declarar 80 categorias")
        save_yaml(output / "source.yaml", {"path": "coco8", "train": native["train"], "val": native["val"],
                                           "names": names})
        evidence["extracted"] = {str(p.relative_to(output)): file_hash(p)
                                 for p in sorted((output / "coco8").rglob("*")) if p.is_file()}
        atomic_json(run.path / "acquisition.json", evidence)
        run.status["artifacts"].update(assets=str(output), evidence=str(run.path / "acquisition.json"))
        return output, classes, evidence


def smoke(config_path: Path, runs_root=None):
    config = read_config(config_path)
    root = runs_root or reference(config_path, config["runs"])
    with Run(root, "yolo26l_smoke", {**config, "recipe_sha256": file_hash(config_path)}) as run:
        assets, classes, evidence = acquire(config_path, config, root)
        recipes = run.path / "recipes"
        source_path = save_yaml(recipes / "source.yaml", {
            "id": "coco8", "format": "yolo", "root": str(assets), "images": ["coco8/images/*/*"],
            "subsets": [{"name": "coco8", "yaml": "source.yaml"}],
            "annotation_scope": {c: "exhaustive" for c in classes},
            "purpose": "software_smoke_only",
            "coverage_basis": "COCO8 publicado para teste de detecção; não é validação de cobertura Tower Vision",
        })
        ingestion = save_yaml(recipes / "ingest.yaml", {"datasets": [str(source_path)],
                              "output": str(run.path / "imported"), "runs": str(root)})
        imported = core("ingest", "--config", ingestion)
        audit = core("audit", "--manifest", imported["manifest"], "--runs-root", root)
        audit_report = load_json(Path(audit["report"]))
        if imported["issues"] or audit_report["issues"] or imported["images"] != 8:
            raise ValueError("COCO8 apresentou inconsistências; consulte a auditoria, sem correção automática")
        manifest, records = read_snapshot(Path(imported["manifest"]))
        if {s: sum(r.original_splits == [{"split": s}] for r in records) for s in ("train", "val")} != {"train": 4, "val": 4}:
            raise ValueError("COCO8 não corresponde aos splits 4/4 publicados")
        if audit_report["exact_pixel_groups"]:
            raise ValueError("COCO8 contém imagens repetidas; revisão necessária")
        taxonomy = save_yaml(recipes / "taxonomy.yaml", {"labels": {c: {} for c in classes},
                                                        "purpose": "software_smoke_only"})
        mapping = save_yaml(recipes / "mapping.yaml", {"datasets": {"coco8": {c: {"target": c} for c in classes}}})
        decisions = {"dataset_version": manifest["version"], "decisions": [
            {"image_id": r.image_id, "image_sha256": r.sha256, "action": "set_group", "group_id": r.image_id,
             "reason": "Preservar as oito imagens distintas e splits do fixture oficial COCO8; uso técnico"}
            for r in records]}
        atomic_json(recipes / "curation.json", decisions)
        preparation = save_yaml(recipes / "prepare.yaml", {
            "manifest": imported["manifest"], "output": str(run.path / "prepared"), "runs": str(root),
            "taxonomy": str(taxonomy), "mapping": str(mapping), "curation": str(recipes / "curation.json"),
            "view": {"id": "coco8-software-smoke", "classes": classes}, "splits": {"mode": "preserve"},
            "exports": [], "purpose": "software_smoke_only",
        })
        prepared = core("prepare", "--config", preparation)
        training_config = save_yaml(recipes / "train.yaml", {
            "manifest": prepared["manifest"], "checkpoint": evidence["weights"]["path"], "runs": str(root),
            "training": config["training"], "purpose": "software_smoke_only",
        })
        trained = run_model("train", training_config)
        prediction_config = save_yaml(recipes / "predict.yaml", {
            "manifest": prepared["manifest"], "checkpoint": trained["checkpoints"]["best"]["path"],
            "runs": str(root), "inference": config["inference"], "split": "val",
        })
        predicted = run_model("predict", prediction_config)
        evaluation = save_yaml(recipes / "evaluate.yaml", {
            "manifest": prepared["manifest"], "runs": str(root), "split": "val", "max_detections": 100,
        })
        metrics = core("evaluate", "--config", evaluation, "--predictions", predicted["predictions"]["path"])
        repeated = core("evaluate", "--config", evaluation, "--predictions", predicted["predictions"]["path"])
        if {k: v for k, v in metrics.items() if k != "run_id"} != {k: v for k, v in repeated.items() if k != "run_id"}:
            raise ValueError("Avaliações repetidas divergentes")
        summary = {"run_id": run.id, "purpose": "software_smoke_only", "ingest": imported, "audit": audit,
                   "prepare": prepared, "train": trained, "predict": predicted, "evaluate": metrics,
                   "reevaluate": repeated}
        atomic_json(run.path / "summary.json", summary)
        run.status["artifacts"]["summary"] = str(run.path / "summary.json")
        return {"run_id": run.id, "summary": str(run.path / "summary.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="yolo26l")
    parser.add_argument("operation", choices=("smoke", "train", "predict"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "smoke":
            if args.manifest or args.checkpoint:
                raise ValueError("smoke utiliza exclusivamente os assets COCO8 configurados")
            result = smoke(args.config.resolve(), args.runs_root.resolve() if args.runs_root else None)
        else:
            result = run_model(args.operation, args.config.resolve(), manifest_path=args.manifest,
                               checkpoint_path=args.checkpoint,
                               runs_root=args.runs_root.resolve() if args.runs_root else None)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
