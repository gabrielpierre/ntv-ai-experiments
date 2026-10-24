import contextlib
import io
import json
import math
from pathlib import Path

from ntv_ai.data.export import coco_document
from ntv_ai.data.views import eligibility
from ntv_ai.storage import file_hash


def evaluate(manifest, records, predictions_path: Path, config):
    try:
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("Instale ntv-ai[evaluation] para avaliar") from exc
    view = manifest.get("view")
    if not view or not view.get("eligible"):
        raise ValueError("Avaliação requer view elegível")
    classes = view["classes"]
    selected_ids = set(view.get("evaluation_image_ids", view["image_ids"]))
    split = config["split"]
    if split not in ("val", "test"):
        raise ValueError("Avaliação v1 requer split val ou test")
    selected = [r for r in records if r.image_id in selected_ids and r.split == split]
    if not selected:
        raise ValueError("Split de avaliação vazio")
    failures = eligibility(selected, classes)
    if failures:
        raise ValueError(f"Referência incompatível: {failures[:3]}")
    if any(r.metadata.get("transform") for r in selected):
        raise ValueError("Referência de avaliação precisa estar nas imagens originais")
    document, ids = coco_document(selected, manifest["sources"], classes)
    loaded = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    if len({p["image_id"] for p in loaded}) != len(loaded):
        raise ValueError("Predições duplicadas por imagem")
    if {p["image_id"] for p in loaded} != set(ids):
        raise ValueError("Predições ausentes ou de imagens fora do split; registre detections: [] para resultados vazios")
    detections = []
    dimensions = {r.image_id: (r.width, r.height) for r in selected}
    for prediction in loaded:
        if prediction.get("status") != "processed":
            raise ValueError("Predição ausente, incompleta ou com falha")
        for d in prediction["detections"]:
            if d["class_id"] not in classes or not math.isfinite(d["score"]) or not 0 <= d["score"] <= 1:
                raise ValueError("Classe ou confiança inválida")
            x1, y1, x2, y2 = d["bbox_xyxy"]
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                raise ValueError("Caixa de predição inválida")
            width, height = dimensions[prediction["image_id"]]
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ValueError("Predição fora da imagem original; corrigir o produtor sem recorte implícito")
            detections.append({"image_id": ids[prediction["image_id"]], "category_id": classes.index(d["class_id"])+1,
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": d["score"]})
    max_dets = config.get("max_detections", 100)
    if not isinstance(max_dets, int) or max_dets < 10:
        raise ValueError("max_detections deve ser inteiro >= 10")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = document
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {**document, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]

    def mean(values):
        valid = values[values >= 0]
        return float(np.mean(valid)) if valid.size else None

    per_class = {}
    for index, name in enumerate(classes):
        per_class[name] = {"mAP50_95": mean(precision[:, :, index, 0, -1]),
                           "AP50": mean(precision[0, :, index, 0, -1]),
                           "AP75": mean(precision[5, :, index, 0, -1]),
                           "AR50_95": mean(recall[:, index, 0, -1]),
                           "recall50": mean(recall[0, index, 0, -1])}
    return {"mAP50_95": mean(precision[:, :, :, 0, -1]), "AP50": mean(precision[0, :, :, 0, -1]),
            "AP75": mean(precision[5, :, :, 0, -1]), "per_class": per_class,
            "AP_by_size": {name: mean(precision[:, :, :, i, -1])
                           for i, name in enumerate(evaluator.params.areaRngLbl) if name != "all"},
            "AR50_95": mean(recall[:, :, 0, -1]),
            "dataset_version": manifest["version"], "predictions_sha256": file_hash(predictions_path),
            "images": len(selected), "protocol": {"evaluator": "COCOeval", "iou": "0.50:0.05:0.95",
                                                "max_detections": max_dets, "split": split,
                                                "area_units": "original_image_pixels_squared",
                                                "recall": "COCO maximum recall at max_detections"}}
