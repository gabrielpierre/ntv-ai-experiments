import json
from pathlib import Path

from ntv_ai.data.adapters import CocoReader, LabelMeReader, VocReader, YoloReader
from ntv_ai.data.audit import geometry_errors

from conftest import image, save_json, save_yaml


def test_coco_preserves_invalid_boxes_missing_labels_and_namespaces(tmp_path):
    image(tmp_path / "train/a.png")
    image(tmp_path / "train/b.png", "red")
    save_json(tmp_path / "data.json", {
        "images": [{"id": 1, "file_name": "train/a.png", "width": 40, "height": 30}],
        "categories": [{"id": 9, "name": "tower"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 9, "bbox": [-1, 0, 11, 12], "custom": "retained"},
                        {"id": 2, "image_id": 999, "category_id": 9, "bbox": [0, 0, 1, 1]}],
    })
    spec = {"id": "a", "images": ["**/*.png"], "annotations": ["data.json"]}
    reader = CocoReader(spec, tmp_path)
    records = reader.read()
    a = next(r for r in records if r.annotations)
    assert a.annotations[0].geometry.coordinates == [-1, 0, 10, 12]
    assert a.annotations[0].original["custom"] == "retained"
    assert a.annotations[0].source["original_id"] == "1"
    assert a.annotations[0].source["record_index"] == 0
    assert "out_of_bounds" in geometry_errors(a.annotations[0], a)
    assert {"annotation_without_image", "image_without_annotations"} <= {i.code for i in reader.issues}
    others = CocoReader({**spec, "id": "b"}, tmp_path).read()
    assert {r.image_id for r in records}.isdisjoint(r.image_id for r in others)


def test_coco_folds_do_not_duplicate_objects(tmp_path):
    image(tmp_path / "a.png")
    doc = {"images": [{"id": 0, "file_name": "a.png", "width": 40, "height": 30}],
           "categories": [{"id": 0, "name": "tower"}],
           "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [1, 2, 3, 4]}]}
    save_json(tmp_path / "release.json", doc)
    for fold in range(5):
        save_json(tmp_path / f"labels_train_{fold}_resize.json", doc)
    rows = CocoReader({"id": "stn", "images": ["*.png"], "annotations": ["release.json"],
                      "folds": ["labels_*.json"]}, tmp_path).read()
    assert len(rows) == len(rows[0].annotations) == 1
    assert len(rows[0].original_splits) == 5


def test_yolo_context_malformed_rows_and_orphan_labels(tmp_path):
    for subset, label, material in (("glass", "1", "vidro"), ("composite", "iso_poly", "polimero")):
        image(tmp_path / subset / "train/images/a.png")
        save_yaml(tmp_path / subset / "data.yaml", {"names": [label]})
        label_path = tmp_path / subset / "train/labels/a.txt"
        label_path.parent.mkdir(parents=True)
        label_path.write_text("0 0.5 0.5 0.5 0.5\n0 NaN 0 1 1\n0 1 2\n")
    (tmp_path / "glass/train/labels/orphan.txt").write_text("0 0 0 1 1")
    reader = YoloReader({"id": "mpid", "images": ["**/images/*.png"],
                         "subsets": [{"name": s, "yaml": f"{s}/data.yaml", "material": m}
                                     for s, m in (("glass", "vidro"), ("composite", "polimero"))]}, tmp_path)
    records = reader.read()
    assert len(records) == 2
    assert {r.metadata["material"] for r in records} == {"vidro", "polimero"}
    assert all(r.annotations[0].geometry.coordinates == [10, 7.5, 30, 22.5] for r in records)
    assert {i.code for i in reader.issues} == {"annotation_parse_error", "label_without_image"}


def test_voc_homonyms_and_explicit_coordinate_convention(tmp_path):
    for subset in ("Normal", "Defective"):
        image(tmp_path / subset / "images/a.jpg")
        p = tmp_path / subset / "labels/a.xml"
        p.parent.mkdir()
        p.write_text("<annotation><filename>a</filename><size><width>40</width><height>30</height></size>"
                     "<object><name>insulator</name><bndbox><xmin>1</xmin><ymin>2</ymin>"
                     "<xmax>11</xmax><ymax>12</ymax></bndbox></object></annotation>")
    spec = {"id": "cplid", "images": ["**/*.jpg"], "coordinate_convention": "native_endpoints_unverified",
            "subsets": [{"name": s, "directory": s, "synthetic": s == "Defective", "annotations": [f"{s}/labels/*.xml"]}
                        for s in ("Normal", "Defective")]}
    reader = VocReader(spec, tmp_path)
    records = reader.read()
    assert not reader.issues
    assert [len(r.annotations) for r in records] == [1, 1]
    assert records[0].annotations[0].geometry.coordinates == [1, 2, 11, 12]
    assert records[0].annotations[0].geometry.convention == "unverified_native_endpoints"
    spec["coordinate_convention"] = "one_based_inclusive"
    assert VocReader(spec, tmp_path).read()[0].annotations[0].geometry.coordinates == [0, 1, 11, 12]


def test_labelme_keeps_polygons_and_void(tmp_path):
    image(tmp_path / "a.png")
    points = [[0, 0], [12, 0], [12, 12], [0, 12]]
    save_json(tmp_path / "a.json", {"imagePath": "a.png", "imageWidth": 40, "imageHeight": 30,
                                  "shapes": [{"label": "void", "shape_type": "polygon", "points": points},
                                             {"label": "Cable", "shape_type": "linestrip", "points": [[0, 0], [10, 10]]}]})
    (tmp_path / "test.txt").write_text("a.png\n")
    reader = LabelMeReader({"id": "ttpla", "images": ["*.png"], "annotations": ["*.json"], "splits": ["*.txt"]}, tmp_path)
    record = reader.read()[0]
    assert record.annotations[0].geometry.coordinates == points
    assert record.annotations[0].geometry.type == "polygon"
    assert record.annotations[0].flags["ignore"]
    assert record.annotations[1].geometry.type == "polyline"
    assert record.original_splits[0]["split"] == "test"


def test_reader_preserves_corrupt_image_and_annotation_failure(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"not an image")
    (tmp_path / "broken.json").write_text("{broken")
    reader = CocoReader({"id": "bad", "images": ["*.png"], "annotations": ["*.json"]}, tmp_path)
    rows = reader.read()
    assert len(rows) == 1
    assert {"image_decode_error", "annotation_file_error"} <= {i.code for i in reader.issues}
