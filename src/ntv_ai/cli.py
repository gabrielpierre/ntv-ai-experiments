import argparse
import json
import sys
from pathlib import Path

from ntv_ai.experiments.config import read_config, reference
from ntv_ai.experiments.runs import Run
from ntv_ai.storage import atomic_json, load_json, read_snapshot, safe_output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ntv-ai")
    sub = p.add_subparsers(dest="operation", required=True)
    annotator = sub.add_parser("annotator", help="Desenvolvimento versionado do anotador; sem execução implícita")
    annotator_actions = annotator.add_subparsers(dest="action", required=True)
    for name in ("prepare", "recipes", "smoke-recipes", "select", "infer", "status", "export", "lofo-plan", "lofo-prepare"):
        action = annotator_actions.add_parser(name)
        action.add_argument("--config", type=Path, required=True)
        if name == "infer":
            action.add_argument("--limit", type=int, help="Piloto limitado; execução em massa exige configuração autorizada")
    models = sub.add_parser("models", help="Usar o pacote de checkpoints baixado do Drive")
    model_actions = models.add_subparsers(dest="action", required=True)
    for action in ("list", "predict", "restore", "evaluate"):
        cmd = model_actions.add_parser(action)
        cmd.add_argument("--package", type=Path, required=True, help="Pasta extraída do pacote")
        if action in ("predict", "restore", "evaluate"):
            cmd.add_argument("--model", required=True, help="Ex.: yolo26-s, dfine-s, rfdetr-nano")
            cmd.add_argument("--protocol", choices=("full640", "tiles1280"), required=True)
            cmd.add_argument("--seed", type=int, required=True)
            cmd.add_argument("--storage-root", type=Path, default=Path.cwd())
        if action == "predict":
            cmd.add_argument("--source", type=Path, required=True, help="Imagem ou pasta de imagens")
            cmd.add_argument("--runs-root", type=Path)
            cmd.add_argument("--device", type=int, default=0)
            cmd.add_argument("--python", type=Path, help="Python do backend; por padrão usa o ambiente Conda")
        if action == "restore":
            cmd.add_argument("--output", type=Path, required=True, help="Nova pasta da execução restaurada")
            cmd.add_argument("--source-root", action="append", default=[], metavar="ID=PASTA",
                             help="Raiz local de uma fonte de dados; pode ser repetido")
        if action == 'evaluate':
            cmd.add_argument('--target-dataset', type=Path, required=True)
            cmd.add_argument('--runs-root', type=Path)
            cmd.add_argument('--device', type=int, default=0)
            cmd.add_argument('--python', type=Path)
            cmd.add_argument('--split', choices=('val', 'test'), default='test')
    for name in ("train", "predict", "smoke", "profile", "measure"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--storage-root", type=Path)
        cmd.add_argument("--runs-root", type=Path)
        if name == "train":
            cmd.add_argument("--resume", type=Path, help="Diretório da execução anterior; cria uma nova execução")
        if name in ("predict", "measure"):
            cmd.add_argument("--checkpoint", type=Path, required=True)
            cmd.add_argument("--split", choices=("val", "test"), default="val")
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--config", type=Path, required=True)
    acquire.add_argument("--storage-root", type=Path, required=True)
    campaigns = sub.add_parser("campaign")
    actions = campaigns.add_subparsers(dest="action", required=True)
    preparation = actions.add_parser("prepare")
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--runs-root", type=Path)
    launch = actions.add_parser("run")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--storage-root", type=Path)
    launch.add_argument("--runs-root", type=Path)
    launch.add_argument("--resume", type=Path)
    launch.add_argument("--catalog", type=Path)
    planning = actions.add_parser('plan')
    planning.add_argument('--config', type=Path, required=True)
    planning.add_argument('--catalog', type=Path, required=True)
    planning.add_argument('--storage-root', type=Path)
    planning.add_argument('--runs-root', type=Path)
    summary = actions.add_parser("summary")
    summary.add_argument("path", type=Path)
    cross = sub.add_parser('cross-predict')
    cross.add_argument('--source-run', type=Path, required=True)
    cross.add_argument('--target-dataset', type=Path, required=True)
    cross.add_argument('--runs-root', type=Path)
    for name in ("ingest", "prepare", "evaluate", "eligibility"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--runs-root", type=Path)
        if name == "evaluate":
            cmd.add_argument("--predictions", type=Path, required=True)
        if name == "prepare":
            cmd.add_argument("--manifest", type=Path, help="Sobrescreve apenas a referência de entrada da receita")
        if name == "eligibility":
            cmd.add_argument("--manifest", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--runs-root", type=Path, default=Path("runs"))
    audit.add_argument("--baseline", type=Path, help="CSV opcional com dataset, images e annotations")
    runs = sub.add_parser("runs")
    actions = runs.add_subparsers(dest="action", required=True)
    show = actions.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--runs-root", type=Path, default=Path("runs"))
    return p


def execute(args) -> dict:
    if args.operation == "annotator":
        from ntv_ai.annotator import data, inference, lofo, selection, training
        if args.action == "prepare":
            return data.freeze(args.config)
        if args.action in ("recipes", "smoke-recipes"):
            return training.recipes(args.config, smoke=args.action == "smoke-recipes")
        if args.action == "select":
            return selection.select(args.config)
        if args.action == "infer":
            return inference.infer(args.config, args.limit)
        if args.action == "status":
            return inference.status(args.config)
        if args.action == "export":
            return inference.export(args.config)
        return lofo.plan(args.config) if args.action == "lofo-plan" else lofo.prepare(args.config)
    if args.operation == "models":
        if args.action == 'evaluate':
            from ntv_ai.pipelines.model_packages import predict_package
            return predict_package(args.package, args.model, args.protocol, args.seed, None,
                                   storage_root=args.storage_root, runs_root=args.runs_root, device=args.device,
                                   python=args.python, evaluation_target=args.target_dataset, split=args.split)
        from ntv_ai.models.package import ModelPackage
        if args.action == "predict":
            from ntv_ai.pipelines.model_packages import predict_package
            return predict_package(args.package, args.model, args.protocol, args.seed, args.source,
                                   storage_root=args.storage_root, runs_root=args.runs_root,
                                   device=args.device, python=args.python)
        package = ModelPackage(args.package)
        if args.action == "list":
            return package.list_models()
        roots = {}
        for item in args.source_root:
            key, separator, value = item.partition("=")
            if not separator or not key or not value or key in roots:
                raise ValueError("Use --source-root ID=PASTA, sem IDs duplicados")
            roots[key] = str(Path(value).resolve())
        row = package.select(args.model, args.protocol, args.seed)
        return package.restore(row, args.output, args.storage_root, roots)
    if args.operation == 'cross-predict':
        from ntv_ai.experiments.cross import cross_predict
        return cross_predict(args.source_run, args.target_dataset, args.runs_root)
    if args.operation in ("train", "predict", "smoke", "profile", "measure"):
        from ntv_ai.pipelines.experiments import run_experiment
        return run_experiment(args.operation, args.config, storage_root=args.storage_root, runs_root=args.runs_root,
                              checkpoint=getattr(args, "checkpoint", None), resume=getattr(args, "resume", None),
                              split=getattr(args, "split", "val"))
    if args.operation == "acquire":
        from ntv_ai.pipelines.acquisition import acquire
        return acquire(args.config, args.storage_root)
    if args.operation == "campaign":
        if args.action == "prepare":
            if read_config(args.config).get("preparation") == "filtered":
                from ntv_ai.pipelines.filtered_data import prepare_filtered
                return prepare_filtered(args.config, args.runs_root)
            from ntv_ai.pipelines.benchmark_data import prepare_benchmark
            return prepare_benchmark(args.config, args.runs_root)
        from ntv_ai.experiments.campaigns import campaign, summarize
        if args.action == "summary":
            if load_json(args.path/'campaign.json').get('schema_version') == 2:
                from ntv_ai.experiments.matrix import summarize_matrix
                return summarize_matrix(args.path)
            return summarize(args.path)
        if args.action == 'plan' or read_config(args.config).get('schema_version') == 2:
            from ntv_ai.experiments.matrix import matrix_campaign
            return matrix_campaign(args.config, args.catalog, args.storage_root, args.runs_root,
                                   getattr(args, 'resume', None), args.action == 'plan')
        return campaign(args.config, args.storage_root, args.runs_root, args.resume)
    if args.operation == "eligibility":
        from ntv_ai.pipelines.eligibility import run_eligibility
        return run_eligibility(args.config, args.manifest, args.runs_root)
    if args.operation == "runs":
        from ntv_ai.storage import inside
        folder = inside(args.runs_root, args.run_id)
        return {name: load_json(folder / f"{name}.json") for name in ("status", "config", "environment")}
    config_path = getattr(args, "config", None)
    config = read_config(config_path) if config_path else {"manifest": str(args.manifest.resolve())}
    if getattr(args, "baseline", None):
        config["baseline"] = str(args.baseline.resolve())
    protected = []
    records = manifest = sources = None
    if args.operation == "ingest":
        from ntv_ai.pipelines.ingest import resolve_sources
        sources = resolve_sources(config_path, config)
        protected = [Path(s["root"]) for s in sources]
        config["resolved_datasets"] = sources
    else:
        manifest_path = getattr(args, "manifest", None)
        if manifest_path is None:
            if not config.get("manifest"):
                raise ValueError("Informe --manifest ou configure uma referência de manifesto")
            manifest_path = reference(config_path, config["manifest"])
        manifest_path = manifest_path.resolve()
        manifest, records = read_snapshot(manifest_path)
        config["manifest"] = str(manifest_path)
        config["input_version"] = manifest["version"]
        protected = [Path(s["root"]) for s in manifest["sources"].values()] + [manifest_path.parent]
    runs_root = args.runs_root or reference(config_path, config.get("runs", "../runs"))
    runs_root = safe_output(runs_root, protected)
    if args.operation == "evaluate":
        config["predictions"] = str(args.predictions.resolve())
    with Run(runs_root, args.operation, config) as run:
        print(f"run_id: {run.id}", flush=True)
        if args.operation == "ingest":
            from ntv_ai.pipelines.ingest import ingest
            return ingest(config_path, config, sources, run)
        if args.operation == "audit":
            from ntv_ai.data.audit import audit_records, reconcile_counts
            report = audit_records(manifest, records)
            report["ingestion_issues"] = load_json(manifest_path.parent / manifest["issues"])
            if args.baseline:
                report["reconciliation"] = reconcile_counts(report, args.baseline)
            atomic_json(run.path / "audit.json", report)
            run.status["artifacts"]["audit"] = str(run.path / "audit.json")
            return {"run_id": run.id, "report": str(run.path / "audit.json"),
                    "datasets": report["datasets"], "issue_counts": report["issue_counts"],
                    "reconciliation": report.get("reconciliation")}
        if args.operation == "prepare":
            from ntv_ai.pipelines.prepare import prepare
            return prepare(config_path, config, manifest, records, run)
        from ntv_ai.evaluation.bbox import evaluate
        metrics = evaluate(manifest, records, args.predictions, config)
        atomic_json(run.path / "metrics.json", metrics)
        run.status["artifacts"]["metrics"] = str(run.path / "metrics.json")
        return {"run_id": run.id, **metrics}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
