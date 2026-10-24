"""Produce native recipes; never launch training as a side effect of preparation."""
from copy import deepcopy
from pathlib import Path

from ntv_ai.annotator.data import CLASSES
from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.recipes import resolve_recipe
from ntv_ai.pipelines.prepare import materialize_tiles
from ntv_ai.storage import atomic_json, file_hash, read_snapshot, write_snapshot


def descriptor(path):
    manifest, _ = read_snapshot(path)
    return {"manifest": {"path": str(path.resolve()), "sha256": file_hash(path)},
            "version": manifest["version"], "view_id": manifest["view"]["id"],
            "classes": manifest["view"]["classes"], "source_roots": {}}


def build_views(manifest_path, output, workers=4, smoke=False):
    manifest, records = read_snapshot(manifest_path)
    active = [deepcopy(r) for r in records if r.split in ("train", "val", "test") and not r.metadata.get("transform")]
    if smoke:
        active = [next(r for r in active if r.split == split) for split in ("train", "val")]
        for r in active:
            r.metadata["software_fixture_only"] = True
    full_view = deepcopy(manifest["view"])
    full_view.update(image_ids=[r.image_id for r in active], evaluation_image_ids=[r.image_id for r in active],
                     tiling={"enabled": False}, purpose="software_smoke_only" if smoke else manifest["view"].get("purpose", "exploratory"))
    lineage = {"operation": "annotator_software_smoke" if smoke else "annotator_views", "parent_version": manifest["version"]}
    if manifest["lineage"].get("operation") == "lofo" and not smoke:
        lineage.update(manifest["lineage"], audit_root=str(Path(manifest_path).parent.parent))
    sources = deepcopy(manifest["sources"])
    full = write_snapshot(output / "full640", active, sources, lineage, [], view=full_view)
    tiling = {"enabled": True, "size": 1280, "overlap": 0.25, "border_policy": "clip"}
    tiles = materialize_tiles(active, sources, CLASSES, tiling, output / "tiles", workers=workers)
    if smoke:
        # Bound optimizer updates. Keep all validation tiles for original-coordinate COCO evaluation.
        tiles = [r for r in tiles if r.split != "train"] + [r for r in tiles if r.split == "train"][:4]
    tile_view = {**full_view, "id": "annotator-tiles1280", "image_ids": [r.image_id for r in tiles], "tiling": tiling}
    tiled = write_snapshot(output / "tiles1280", active + tiles, sources, lineage, [], view=tile_view)
    return {"full640": full, "tiles1280": tiled}


def recipes(config_path, smoke=False):
    cfg = read_config(config_path)
    root, output = Path(cfg["repository"]), Path(cfg["output"])
    if output.exists():
        raise FileExistsError(output)
    manifests = build_views(Path(cfg["manifest"]), output / "datasets", cfg.get("tile_workers", 4), smoke)
    return write_recipes(cfg, manifests, output, smoke)


def write_recipes(cfg, manifests, output, smoke=False):
    root = Path(cfg["repository"])
    entries = []
    for model in cfg.get("models", ["yolo26-s", "dfine-s", "rfdetr-nano"]):
        for protocol in ("full640", "tiles1280"):
            value = resolve_recipe(root / f"configs/benchmark/{model}-{protocol}.yaml", root)
            value.pop("provenance", None)
            value["dataset"] = descriptor(manifests[protocol])
            value["purpose"] = "software_smoke_only" if smoke else cfg.get("purpose", "exploratory")
            value["training"].update(epochs=1 if smoke else cfg.get("epochs", 100), seed=cfg.get("seed", 43))
            environment = value["runtime"]["environment"]
            value["runtime"].update(python=cfg["interpreters"][environment], cpu_threads=cfg.get("cpu_threads", 4))
            weights = root / value["model"]["weights"]["path"]
            if file_hash(weights) != value["model"]["weights"]["sha256"]:
                raise ValueError(f"Official initialization hash mismatch: {weights}")
            seeds = [cfg.get("seed", 43)] if smoke else cfg.get("seeds", [cfg.get("seed", 43)])
            for seed in seeds:
                value["training"]["seed"] = seed
                suffix = f"-seed{seed}" if len(seeds) > 1 else ""
                path = output / "recipes" / f"{model}-{protocol}{suffix}.json"
                atomic_json(path, value)
                entries.append({"model": model, "protocol": protocol, "seed": seed, "recipe": str(path),
                            "official_weights_sha256": value["model"]["weights"]["sha256"],
                            "purpose": value["purpose"]})
    atomic_json(output / "jobs.json", {"jobs": entries, "automatic_training": False,
                                      "automatic_mass_inference": False})
    return {"jobs": str(output / "jobs.json"), "count": len(entries), "executed": False}
