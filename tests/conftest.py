import json
from pathlib import Path

import pytest
from PIL import Image
import yaml

from ntv_ai.cli import execute, parser
from ntv_ai.storage import read_snapshot


def save_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value))
    return path


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def image(path, color="white", size=(40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    images, annotations = [], []
    for i, (split, color) in enumerate((("train", "red"), ("val", "blue"), ("test", "green")), 1):
        image(raw / split / f"{i}.png", color)
        images.append({"id": i, "file_name": f"{split}/{i}.png", "width": 40, "height": 30})
        annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": [4, 5, 10, 12]})
    save_json(raw / "annotations.json", {"images": images, "annotations": annotations,
                                        "categories": [{"id": 1, "name": "native"}]})
    save_yaml(tmp_path / "source.yaml", {"id": "fixture", "format": "coco", "root": "raw",
                                       "images": ["**/*.png"], "annotations": ["annotations.json"],
                                       "annotation_scope": {"isolador": "exhaustive"}})
    ingest = save_yaml(tmp_path / "ingest.yaml", {"datasets": ["source.yaml"], "output": "derived", "runs": "runs"})
    result = execute(parser().parse_args(["ingest", "--config", str(ingest)]))
    manifest_path = Path(result["manifest"])
    manifest, records = read_snapshot(manifest_path)
    save_yaml(tmp_path / "taxonomy.yaml", {"labels": {"isolador": {"attributes": {"material": ["vidro"]}}}})
    save_yaml(tmp_path / "mapping.yaml", {"datasets": {"fixture": {"native": {"target": "isolador"}}}})
    decisions = [{"image_id": r.image_id, "image_sha256": r.sha256, "action": "set_group",
                  "group_id": f"flight-{i}", "reason": "Synthetic independent captures"} for i, r in enumerate(records)]
    save_json(tmp_path / "curation.json", {"dataset_version": manifest["version"], "decisions": decisions})
    preparation = {"manifest": str(manifest_path), "output": "prepared", "runs": "runs",
                   "taxonomy": "taxonomy.yaml", "mapping": "mapping.yaml", "curation": "curation.json",
                   "view": {"id": "bbox-v1", "classes": ["isolador"]},
                   "splits": {"mode": "preserve"}, "exports": ["coco", "yolo"]}
    prepare_path = save_yaml(tmp_path / "prepare.yaml", preparation)
    return {"root": tmp_path, "raw": raw, "manifest": manifest_path,
            "prepare": prepare_path, "config": preparation, "records": records}
