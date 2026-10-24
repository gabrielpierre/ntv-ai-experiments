import math
import time
from pathlib import Path

from PIL import Image

from ntv_ai.domain.records import Detection, PredictionRecord
from ntv_ai.storage import file_hash, inside


def detections(boxes, scores, labels, classes, size, threshold, maximum, clip=False):
    width, height = size
    result = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        if not math.isfinite(float(label)) or int(label) != label or not 0 <= label < len(classes):
            raise ValueError("Classe inválida retornada pelo backend")
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in [*box, score]) or not 0 <= score <= 1:
            raise ValueError("Detecção não finita ou confiança inválida")
        box = list(map(float, box))
        if score < threshold:
            continue
        if clip:
            box = [min(max(x, 0), size[i % 2]) for i, x in enumerate(box)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError("Caixa fora da amostra recebida")
        result.append(Detection(classes[int(label)], box, float(score)))
    return sorted(result, key=lambda d: -d.score)[:maximum]


def predict_samples(predictor, records, sources):
    import torch
    predictions, timings = [], []
    for record in records:
        path = inside(Path(sources[record.dataset_id]["root"]), record.path)
        if file_hash(path) != record.sha256:
            raise ValueError(f"Imagem alterada: {record.image_id}")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with Image.open(path) as image:
            if image.size != (record.width, record.height):
                raise ValueError("Dimensões divergentes")
            items, model_seconds = predictor(image.convert("RGB"))
        torch.cuda.synchronize()
        predictions.append(PredictionRecord(record.image_id, items))
        timings.append({"image_id": record.image_id, "model_seconds": model_seconds,
                        "sample_seconds": time.perf_counter() - start})
    return predictions, timings


def predict_image(predictor, image, tiling, tile_iou):
    """Predict a new original image, sharing the benchmark geometry and NMS."""
    from ntv_ai.transforms.geometry import nms, reproject_box, tile_windows
    if not tiling.get("tiling"):
        items, seconds = predictor(image)
        return items, {"samples": 1, "model_seconds": seconds}
    items, seconds, count = [], 0.0, 0
    for window in tile_windows(image.width, image.height, tiling["size"], tiling["overlap"]):
        local, elapsed = predictor(image.crop(window))
        transform = {"window": window, "scale": [1, 1], "padding": [0, 0]}
        items.extend(Detection(d.class_id, reproject_box(d.bbox_xyxy, transform), d.score) for d in local)
        seconds += elapsed
        count += 1
    return nms(items, tile_iou), {"samples": count, "model_seconds": seconds}
