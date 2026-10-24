import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

from ntv_ai.experiments.config import read_config
from ntv_ai.models.common import artifact
from ntv_ai.storage import atomic_json, file_hash, inside, load_json, read_snapshot


def acquire(catalog_path, storage_root):
    from ntv_ai.pipelines.yolo26l import download
    root = Path(storage_root).resolve()
    catalog = read_config(Path(catalog_path))
    evidence = {"catalog_sha256": file_hash(Path(catalog_path)), "files": [], "repositories": []}
    for item in catalog["files"]:
        path = inside(root, item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            download(item, path)
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Hash divergente: {path}")
        evidence["files"].append({**item, "verified": True})
    for item in catalog.get("repositories", []):
        path = inside(root, item["path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", item["url"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", item["revision"]], check=True)
        actual = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        if actual != item["revision"] or dirty:
            raise ValueError("Código externo divergente; aquisição não sobrescreve alterações locais")
        evidence["repositories"].append({**item, "verified": True})
    atomic_json(root / "data/assets/acquisition.json", evidence, replace=True)
    fixture = prepare_coco8(root)
    return {"acquisition": str(root / "data/assets/acquisition.json"), "dataset": str(fixture)}


def prepare_coco8(root):
    from ntv_ai.pipelines.yolo26l import core, save_yaml
    fixture = root / "data/fixtures/coco8"
    descriptor = fixture / "dataset.json"
    if descriptor.exists():
        data = load_json(descriptor)
        if file_hash(root / data["manifest"]["path"]) != data["manifest"]["sha256"]:
            raise ValueError("Fixture preparado foi modificado")
        return descriptor
    fixture.mkdir(parents=True, exist_ok=True)
    raw = fixture / "raw"
    raw.mkdir(exist_ok=True)
    with zipfile.ZipFile(root / "data/assets/coco8.zip") as archive:
        for name in archive.namelist():
            inside(raw, name)
        archive.extractall(raw)
    native = read_config(root / "data/assets/coco8.yaml")
    classes = [native["names"][i] for i in range(80)]
    source = raw / "source.yaml"
    if not source.exists():
        save_yaml(source, {"path": "coco8", "train": native["train"], "val": native["val"], "names": native["names"]})
    # Each preparation attempt has its own immutable directory, including failed attempts.
    from ntv_ai.experiments.runs import Run
    with Run(root / "runs", "prepare_software_fixture", {"fixture": "coco8", "purpose": "software_smoke_only"}) as run:
        recipes = run.path / "recipes"
        source_path = save_yaml(recipes / "source.yaml", {
            "id": "coco8", "format": "yolo", "root": str(raw), "images": ["coco8/images/*/*"],
            "subsets": [{"name": "coco8", "yaml": "source.yaml"}],
            "annotation_scope": {c: "exhaustive" for c in classes}, "purpose": "software_smoke_only",
            "coverage_basis": "Fixture oficial COCO8; não representa cobertura Tower Vision"})
        ingestion = save_yaml(recipes / "ingest.yaml", {"datasets": [str(source_path)],
                              "output": str(fixture / "imported"), "runs": str(root / "runs")})
        imported = core("ingest", "--config", ingestion)
        audited = core("audit", "--manifest", imported["manifest"], "--runs-root", root / "runs")
        audit = load_json(Path(audited["report"]))
        manifest, records = read_snapshot(Path(imported["manifest"]))
        if imported["images"] != 8 or imported["issues"] or audit["issues"] or audit["exact_pixel_groups"]:
            raise ValueError("Fixture COCO8 divergente; nenhuma correção automática")
        counts = {s: sum(r.original_splits == [{"split": s}] for r in records) for s in ("train", "val")}
        if counts != {"train": 4, "val": 4}:
            raise ValueError("COCO8 requer splits 4/4")
        taxonomy = save_yaml(recipes / "taxonomy.yaml", {"labels": {c: {} for c in classes}})
        mapping = save_yaml(recipes / "mapping.yaml", {"datasets": {"coco8": {c: {"target": c} for c in classes}}})
        atomic_json(recipes / "curation.json", {"dataset_version": manifest["version"], "decisions": [
            {"image_id": r.image_id, "image_sha256": r.sha256, "action": "set_group", "group_id": r.image_id,
             "reason": "Fixture técnico: preservar oito imagens distintas e splits oficiais COCO8"} for r in records]})
        prep = save_yaml(recipes / "prepare.yaml", {"manifest": imported["manifest"],
            "output": str(fixture / "prepared"), "runs": str(root / "runs"), "taxonomy": str(taxonomy),
            "mapping": str(mapping), "curation": str(recipes / "curation.json"),
            "view": {"id": "coco8-software-smoke", "classes": classes}, "splits": {"mode": "preserve"},
            "exports": [], "purpose": "software_smoke_only"})
        result = core("prepare", "--config", prep)
        path = Path(result["manifest"])
        manifest, _ = read_snapshot(path)
        ref = asdict(artifact(path))
        ref["path"] = str(path.relative_to(root))
        atomic_json(descriptor, {"manifest": ref, "version": manifest["version"], "view_id": manifest["view"]["id"],
                    "classes": classes, "source_roots": {"coco8": str(raw.relative_to(root))}})
    return descriptor
