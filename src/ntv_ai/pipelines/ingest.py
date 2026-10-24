from dataclasses import asdict
from pathlib import Path

from ntv_ai.data.adapters import READERS
from ntv_ai.experiments.config import read_config, reference
from ntv_ai.storage import digest, safe_output, write_snapshot


def resolve_sources(config_path: Path, config: dict) -> list[dict]:
    sources = []
    for name in config["datasets"]:
        path = reference(config_path, name)
        spec = read_config(path)
        spec["root"] = str(reference(path, spec["root"]))
        sources.append(spec)
    ids = [s["id"] for s in sources]
    if not ids:
        raise ValueError("Nenhuma fonte configurada")
    if len(set(ids)) != len(ids):
        raise ValueError("IDs de datasets repetidos")
    return sources


def ingest(config_path: Path, config: dict, sources: list[dict], run) -> dict:
    roots = [Path(s["root"]) for s in sources]
    output = safe_output(reference(config_path, config["output"]) / run.id, roots)
    records, issues, metadata = [], [], {}
    for spec in sources:
        run.logger.info("Importando %s", spec["id"])
        reader = READERS[spec["format"]](spec, Path(spec["root"]))
        source_records = reader.read()
        records.extend(source_records)
        issues.extend(asdict(i) for i in reader.issues)
        config_identity = {k: v for k, v in spec.items() if k != "root"}
        metadata[spec["id"]] = {
            "root": spec["root"], "config": config_identity, "evidence": reader.evidence,
            "version": digest({"config": config_identity, "evidence": reader.evidence,
                               "images": {r.path: r.sha256 for r in source_records}}),
        }
        print(f'{spec["id"]}: {len(source_records)} imagens, {sum(len(r.annotations) for r in source_records)} anotações', flush=True)
    manifest_path = write_snapshot(output, records, metadata, {"operation": "ingest"}, issues)
    run.status["artifacts"]["manifest"] = str(manifest_path)
    return {"run_id": run.id, "manifest": str(manifest_path), "images": len(records),
            "annotations": sum(len(r.annotations) for r in records), "issues": len(issues)}
