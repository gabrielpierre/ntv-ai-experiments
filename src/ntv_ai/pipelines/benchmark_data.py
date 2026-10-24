"""Explicit, immutable preparation of the reviewed NTV pilot and external candidates."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ntv_ai.data.adapters import READERS
from ntv_ai.data.groups import group_records, historical_links
from ntv_ai.data.harmonize import harmonize
from ntv_ai.data.views import eligibility
from ntv_ai.experiments.config import read_config
from ntv_ai.experiments.runs import Run
from ntv_ai.pipelines.prepare import materialize_tiles
from ntv_ai.storage import atomic_json, digest, file_hash, load_json, read_snapshot, write_snapshot


def tile_impact(originals, tiles, classes):
    from ntv_ai.transforms.geometry import clip_box
    windows = {}
    for tile in tiles:
        transform = tile.metadata['transform']
        windows.setdefault(transform['parent_image_id'], []).append(transform['window'])
    rows = []
    for image in originals:
        for annotation in image.annotations:
            if annotation.label not in classes or annotation.mapping_status != 'mapped':
                continue
            x1,y1,x2,y2 = annotation.geometry.coordinates
            area = (x2-x1)*(y2-y1)
            fractions=[]
            for window in windows[image.image_id]:
                clipped=clip_box(annotation.geometry.coordinates,window)
                fractions.append((clipped[2]-clipped[0])*(clipped[3]-clipped[1])/area if clipped else 0)
            rows.append({'image_id':image.image_id,'annotation_id':annotation.annotation_id,
                         'class':annotation.label,'split':image.split,'area':area,
                         'max_single_tile_iou':max(fractions)})
    return {'note':'Geometric upper bound after clipping; NMS does not join box fragments. Not model metrics.',
            'negative_tiles':sum(not t.annotations for t in tiles), 'tiles':len(tiles),
            'by_class':{c:{'objects':sum(r['class']==c for r in rows),
                          'cannot_reach_iou50':sum(r['class']==c and r['max_single_tile_iou']<.5 for r in rows),
                          'large_objects':sum(r['class']==c and r['area']>96**2 for r in rows)} for c in classes},
            'objects':rows}


def temporal_pilot(records, expected_images=50):
    records = deepcopy(records)
    rows = sorted(records, key=lambda r: Path(r.path).name)
    if len(rows) != expected_images or expected_images != 50:
        raise ValueError("O split aprovado requer o lote exato de 50 imagens")
    for i, r in enumerate(rows):
        split = "train" if i < 30 else "val" if 32 <= i < 40 else "test" if i >= 42 else None
        r.split = split
        r.group_id = f"ntv-pilot-{split or 'embargo'}"
        r.metadata.update(grouping_reviewed=True, capture_id="DJI_20250911_104934_105524",
                          independence="within_capture_exploratory", temporal_index=i,
                          excluded=split is None, exclusion_reason="temporal_embargo" if split is None else None)
    seen = {}
    for r in rows:
        if r.split:
            for fingerprint in (r.sha256, r.pixel_sha256):
                if fingerprint and fingerprint in seen and seen[fingerprint] != r.split:
                    raise ValueError("Duplicata exata atravessa partições do piloto")
                seen[fingerprint] = r.split
    return rows


def descriptor(path):
    manifest, _ = read_snapshot(path)
    return {"manifest": {"path": str(path.resolve()), "sha256": file_hash(path)},
            "version": manifest["version"], "view_id": manifest["view"]["id"],
            "classes": manifest["view"]["classes"], "source_roots": {}}


def prepare_benchmark(config_path, runs_root=None):
    config_path = Path(config_path).resolve()
    config = read_config(config_path)
    root = (config_path.parent / config["storage_root"]).resolve()
    with Run(Path(runs_root or root / "runs"), "benchmark_prepare", config) as run:
        taxonomy = read_config(root / config["taxonomy"])
        mapping = read_config(root / config["mapping"])
        ntv_spec = read_config(root / config["ntv_source"])
        ntv_spec["root"] = str(root / ntv_spec["root"])
        reader = READERS[ntv_spec["format"]](ntv_spec, Path(ntv_spec["root"]))
        ntv = reader.read()
        if reader.issues:
            raise ValueError(f"Importação NTV-AI com pendências: {[asdict(x) for x in reader.issues[:5]]}")
        raw_identity = {r.path: r.sha256 for r in ntv}
        raw_identity.update(reader.evidence)
        if digest(raw_identity) != config["reviewed_input_digest"]:
            raise ValueError("Confirmação de cobertura pertence a outro lote")
        if len(ntv) != 50 or sum(len(r.annotations) for r in ntv) != 968:
            raise ValueError("Lote NTV-AI diverge das contagens aprovadas")
        sources = {"ntv_ai": {"root": ntv_spec["root"], "evidence": reader.evidence,
                              "version": digest(raw_identity),
                              "config": {k: v for k, v in ntv_spec.items() if k != "root"}}}
        ntv = harmonize(ntv, taxonomy, mapping)
        classes = config["ntv_classes"]
        for r in ntv:
            r.annotation_scope = dict.fromkeys(classes, "exhaustive")
            r.metadata["coverage_confirmation"] = config["coverage_confirmation"]
        ntv = temporal_pilot(ntv)
        external_path = root / config["external_manifest"]["path"]
        if file_hash(external_path) != config["external_manifest"]["sha256"]:
            raise ValueError("Manifesto externo divergente")
        external_manifest, external = read_snapshot(external_path)
        # No historical snapshot is rewritten; annotations are harmonized into a new version.
        external = harmonize(external, taxonomy, mapping)
        external_sources = deepcopy(external_manifest["sources"])
        for name, source in external_sources.items():
            source["root"] = str(root / "data/external" / name)
        sources.update(external_sources)
        all_records = external + ntv
        links = historical_links(external,
            [root / p for p in config["historical_inventories"]], root / config["historical_review"])
        group_records(all_records, links)
        # A repeated capture in another base must not be considered an independent target.
        locations = {}
        for r in all_records:
            if not r.metadata.get("excluded"):
                for fingerprint in (r.sha256, r.pixel_sha256):
                    if fingerprint:
                        locations.setdefault(fingerprint, set()).add((r.dataset_id, r.split))
        ntv_fingerprints = {x for r in ntv if r.split for x in (r.sha256, r.pixel_sha256) if x}
        if any(len({d for d, _ in locations[x]}) > 1 for x in ntv_fingerprints):
            raise ValueError("Sobreposição exata NTV-AI/externas requer revisão de grupos")
        catalog = {"purpose": "exploratory", "datasets": {}, "preparation_run": run.id,
                   "storage_root": str(root),
                   "mapping_sha256": digest(mapping), "taxonomy_sha256": digest(taxonomy)}
        lineage = {"operation": "benchmark_prepare", "external_parent": external_manifest["version"],
                   "ntv_input_digest": config["reviewed_input_digest"],
                   "mapping_sha256": digest(mapping), "taxonomy_sha256": digest(taxonomy),
                   "coverage_confirmation": config["coverage_confirmation"],
                   "independence": "within_capture_exploratory"}
        output = root / "data/prepared" / run.id
        source_errors = [i for i in load_json(external_path.parent / external_manifest['issues'])
                         if i.get('code') in ('annotation_file_error', 'annotation_without_image',
                            'image_reference_unresolved', 'label_without_image', 'fold_parse_error')]
        for name in config["datasets"]:
            rows = [r for r in all_records if r.dataset_id == name]
            active = [r for r in rows if not r.metadata.get("excluded")]
            task_classes = classes if name == "ntv_ai" else config["external_classes"][name]
            problems = eligibility(active, task_classes)
            if name != "ntv_ai" and source_errors:
                problems.append({"codes": ["unresolved_source_issue"], "count": len(source_errors)})
            catalog["datasets"][name] = {}
            for protocol in ("full640", "tiles1280"):
                transform = {"enabled": protocol == "tiles1280", "size": 1280, "overlap": .25, "border_policy": "clip"}
                snapshot_sources = {name: deepcopy(sources[name])}
                sample_rows = active
                stored = deepcopy(rows)
                folder = output / name / protocol
                if transform["enabled"] and not problems:
                    sample_rows = materialize_tiles(active, snapshot_sources, task_classes, transform,
                                                    output / name / "tiles")
                    stored.extend(sample_rows)
                view = {"id": f"{name}-{protocol}-v1", "classes": task_classes,
                        "image_ids": [r.image_id for r in sample_rows],
                        "evaluation_image_ids": [r.image_id for r in active], "eligible": not problems,
                        "purpose": "exploratory", "splits": {"mode": "explicit"}, "tiling": transform,
                        "independence": "within_capture_exploratory" if name == "ntv_ai" else "unreviewed"}
                manifest = write_snapshot(folder, stored, snapshot_sources, lineage, problems, view=view)
                if transform['enabled'] and not problems:
                    atomic_json(folder/'tile-impact.json', tile_impact(active,sample_rows,task_classes))
                atomic_json(folder / "splits.json", {s: [r.image_id for r in active if r.split == s] for s in ("train", "val", "test")})
                entry = {**descriptor(manifest), "eligible": not problems, "blockers": problems,
                         "counts": dict(Counter(r.split or 'unassigned' for r in active))}
                atomic_json(folder / 'dataset.json', descriptor(manifest))
                catalog["datasets"][name][protocol] = entry
        atomic_json(run.path / "datasets.json", catalog)
        run.status["artifacts"]["datasets"] = str(run.path / "datasets.json")
        return {"run_id": run.id, "catalog": str(run.path / "datasets.json"),
                "datasets": {k: {p: v['eligible'] for p,v in d.items()} for k,d in catalog['datasets'].items()}}
