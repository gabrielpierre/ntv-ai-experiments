import hashlib
from copy import deepcopy
from pathlib import Path

from PIL import Image

from ntv_ai.data.audit import audit_records
from ntv_ai.data.export import export_views
from ntv_ai.data.groups import assign_splits, group_records, historical_links
from ntv_ai.data.harmonize import curate, harmonize
from ntv_ai.data.views import eligibility, view_records
from ntv_ai.experiments.config import read_config, reference
from ntv_ai.storage import atomic_json, digest, file_hash, inside, load_json, safe_output, write_snapshot
from ntv_ai.transforms.geometry import crop_record, tile_windows


def materialize_tiles(records, sources, classes, config, output, workers=1):
    if config["border_policy"] != "clip":
        raise ValueError("Tiling v1 requer border_policy: clip")
    output.mkdir(parents=True, exist_ok=False)
    def materialize_image(r):
        result = []
        image_path = inside(Path(sources[r.dataset_id]["root"]), r.path)
        source = deepcopy(r)
        source.annotations = [a for a in source.annotations if a.mapping_status == "mapped" and a.label in classes]
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            for window in tile_windows(r.width, r.height, config["size"], config["overlap"]):
                tile = crop_record(source, window)
                tile.metadata["parent_dataset_id"] = r.dataset_id
                tile.dataset_id = "tiles:" + r.dataset_id
                tile.path = tile.image_id + ".png"
                cropped = im.crop(window)
                cropped.save(output / tile.path)
                tile.sha256 = file_hash(output / tile.path)
                h = hashlib.sha256(str(cropped.size).encode())
                h.update(cropped.tobytes())
                tile.pixel_sha256 = h.hexdigest()
                result.append(tile)
        return result
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tiles = [tile for batch in pool.map(materialize_image, records) for tile in batch]
    for dataset in {r.dataset_id for r in tiles}:
        members = [r for r in tiles if r.dataset_id == dataset]
        sources[dataset] = {"root": str(output.resolve()), "evidence": {},
                            "version": digest({r.path: r.sha256 for r in members}),
                            "config": {"transform": config}}
    return tiles


def prepare(config_path, config, manifest, records, run):
    protected = [Path(s["root"]) for s in manifest["sources"].values()] + [Path(config["manifest"]).parent]
    output = safe_output(reference(config_path, config["output"]) / run.id, protected)
    taxonomy = read_config(reference(config_path, config["taxonomy"]))
    mapping = read_config(reference(config_path, config["mapping"]))
    resolved = {"taxonomy": taxonomy, "mapping": mapping}
    atomic_json(run.path / "resolved_preparation.json", resolved)
    integrity = audit_records(manifest, records)
    mismatches = [i for i in integrity["issues"] if i["code"].endswith("integrity_mismatch")]
    if mismatches:
        atomic_json(run.path / "integrity_failures.json", mismatches)
        raise ValueError("Fontes alteradas desde a importação")
    prepared = harmonize(records, taxonomy, mapping)
    lineage = {"operation": "prepare", "parent_version": manifest["version"],
               "taxonomy_sha256": digest(taxonomy), "mapping_sha256": digest(mapping)}
    if config.get("curation"):
        decisions = load_json(reference(config_path, config["curation"]))
        prepared = curate(prepared, decisions, manifest["version"])
        lineage["curation_sha256"] = digest(decisions)
        atomic_json(run.path / "curation.json", decisions)
    links = []
    if config.get("historical_review"):
        spec = config["historical_review"]
        inventories = [reference(config_path, p) for p in spec["inventories"]]
        review = reference(config_path, spec["review"])
        links = historical_links(prepared, inventories, review)
        lineage["historical_evidence"] = {str(p): file_hash(p) for p in [*inventories, review]}
    group_records(prepared, links)
    view_config = config["view"]
    selected = view_records(prepared, view_config)
    split_issues = assign_splits(selected, config.get("splits", {}))
    classes = view_config["classes"]
    defined = set(taxonomy["labels"]) | set(taxonomy.get("coarse_labels", {}))
    if set(classes) - defined:
        raise ValueError("A view contém conceitos fora da taxonomia")
    problems = eligibility(selected, classes)
    input_issues = load_json(Path(config["manifest"]).parent / manifest["issues"])
    global_failures = [i for i in input_issues if i["code"] in (
        "annotation_file_error", "annotation_without_image", "image_reference_unresolved", "label_without_image", "fold_parse_error")]
    problems.extend({"codes": ["unresolved_source_issue"], "source_issue": i} for i in global_failures)
    problems.extend(split_issues)
    sources = deepcopy(manifest["sources"])
    evaluation_ids = [r.image_id for r in selected]
    transform_config = config.get("tiling", {"enabled": False})
    if transform_config.get("enabled") and not problems:
        tile_output = safe_output(output.with_name(output.name + "-tiles"), protected)
        selected = materialize_tiles(selected, sources, classes, transform_config, tile_output)
        prepared.extend(selected)
    view = {"id": view_config["id"], "classes": classes, "image_ids": [r.image_id for r in selected],
            "evaluation_image_ids": evaluation_ids, "eligible": not problems,
            "splits": config.get("splits", {}), "tiling": transform_config}
    lineage["recipe_sha256"] = digest({k: v for k, v in config.items() if k not in ("output", "runs", "manifest")})
    path = write_snapshot(output, prepared, sources, lineage, problems, view=view)
    run.status["artifacts"]["manifest"] = str(path)
    atomic_json(run.path / "eligibility.json", {"eligible": not problems, "problems": problems})
    atomic_json(output / "splits.json", {s: sorted(r.image_id for r in selected if r.split == s) for s in ("train", "val", "test")})
    atomic_json(output / "groups.json", {r.image_id: r.group_id for r in prepared})
    formats = config.get("exports", [])
    if formats:
        if problems:
            raise ValueError(f"Export bloqueado: {len(problems)} pendências; consulte {run.path / 'eligibility.json'}")
        exports = safe_output(output / "exports", protected)
        export_views(selected, sources, classes, exports, formats)
        run.status["artifacts"]["exports"] = str(exports)
    return {"run_id": run.id, "manifest": str(path), "eligible": not problems,
            "issues": len(problems), "images_in_view": len(selected)}
