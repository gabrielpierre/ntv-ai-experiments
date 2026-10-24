import hashlib
import json
import os
import tempfile
from pathlib import Path

from .domain.records import SCHEMA_VERSION, ImageRecord


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_output(path: Path, protected: list[Path]) -> Path:
    result = path.resolve()
    for original in protected:
        base = original.resolve()
        if result == base or result.is_relative_to(base):
            raise ValueError(f"Saída dentro de fonte protegida: {result}")
    return result


def atomic_json(path: Path, value: object, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Referência fora da fonte: {relative}")
    return path


def write_snapshot(
    output: Path, records: list[ImageRecord], sources: dict, lineage: dict, issues: list[dict],
    *, view: dict | None = None,
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    lines = b"".join(json_bytes(r.to_dict()) for r in sorted(records, key=lambda r: r.image_id))
    records_hash = hashlib.sha256(lines).hexdigest()
    atomic_json(output / "issues.json", issues)
    record_file = output / "images.jsonl"
    with record_file.open("xb") as stream:
        stream.write(lines)
        stream.flush()
        os.fsync(stream.fileno())
    identity = {
        "schema_version": SCHEMA_VERSION,
        "records_sha256": records_hash,
        "sources": {k: {n: v for n, v in s.items() if n != "root"} for k, s in sources.items()},
        "lineage": lineage,
        "view": view,
        "issues_sha256": file_hash(output / "issues.json"),
    }
    manifest = {
        **identity, "version": digest(identity), "sources": sources, "records": "images.jsonl",
        "issues": "issues.json", "images": len(records),
        "annotations": sum(len(r.annotations) for r in records),
        "training_ready": bool(view and view.get("eligible")),
    }
    atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def read_snapshot(path: Path) -> tuple[dict, list[ImageRecord]]:
    manifest = load_json(path)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Versão de esquema não suportada")
    records_path = inside(path.parent, manifest["records"])
    if file_hash(records_path) != manifest["records_sha256"]:
        raise ValueError("Manifesto alterado: hash dos registros divergente")
    issues_path = inside(path.parent, manifest["issues"])
    if file_hash(issues_path) != manifest["issues_sha256"]:
        raise ValueError("Registro de pendências alterado")
    identity = {k: manifest[k] for k in ("schema_version", "records_sha256", "lineage", "view", "issues_sha256")}
    identity["sources"] = {
        k: {n: v for n, v in source.items() if n != "root"} for k, source in manifest["sources"].items()
    }
    if digest(identity) != manifest["version"]:
        raise ValueError("Identidade do manifesto divergente")
    records = [ImageRecord.from_dict(json.loads(line)) for line in records_path.read_text().splitlines()]
    if len({r.image_id for r in records}) != len(records):
        raise ValueError("image_id duplicado no manifesto")
    if len(records) != manifest["images"] or sum(len(r.annotations) for r in records) != manifest["annotations"]:
        raise ValueError("Contagens do manifesto divergentes")
    if manifest["training_ready"] != bool(manifest["view"] and manifest["view"].get("eligible")):
        raise ValueError("Estado de elegibilidade divergente")
    return manifest, records
