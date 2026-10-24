from pathlib import Path

from ntv_ai.data.export import coco_document, export_views
from ntv_ai.storage import atomic_json


def export_dataset(records, sources, classes, output: Path, format: str):
    if format == "yolo":
        export_views(records, sources, classes, output, ["yolo"])
        return output / "yolo" / "data.yaml"
    if format not in ("coco_zero_based", "roboflow_coco"):
        raise ValueError(f"Export desconhecido: {format}")
    output.mkdir(parents=True, exist_ok=False)
    # Both integrations use contiguous zero-based categories. RF-DETR 1.11 explicitly
    # remaps COCO categories; no fake background/supercategory is exported.
    atomic_json(output / "class_map.json", [
        {"canonical": name, "coco_id": i + 1, "native_id": i} for i, name in enumerate(classes)
    ])
    for split in ("train", "val", "test"):
        document, _ = coco_document([r for r in records if r.split == split], sources, classes)
        for category in document["categories"]:
            category["id"] -= 1
            category["supercategory"] = "object"
        for annotation in document["annotations"]:
            annotation["category_id"] -= 1
        if format == "roboflow_coco":
            folder = output / ("valid" if split == "val" else split)
            folder.mkdir()
            for image in document["images"]:
                original = Path(image["file_name"])
                name = image["canonical_image_id"] + original.suffix
                (folder / name).symlink_to(original)
                image["file_name"] = name
            path = folder / "_annotations.coco.json"
        else:
            path = output / f"{split}.json"
        atomic_json(path, document)
    return output
