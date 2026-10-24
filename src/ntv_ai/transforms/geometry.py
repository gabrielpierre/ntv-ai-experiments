from copy import deepcopy

from ntv_ai.domain import Detection
from ntv_ai.storage import digest


def tile_windows(width: int, height: int, size: int, overlap: float):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Tamanho/overlap inválidos")
    step = max(1, int(size * (1-overlap)))

    def starts(n):
        if n <= size:
            return [0]
        positions = list(range(0, n-size+1, step))
        if positions[-1] != n-size:
            positions.append(n-size)
        return positions

    return [(x, y, min(x+size, width), min(y+size, height)) for y in starts(height) for x in starts(width)]


def clip_box(box, window):
    x1, y1, x2, y2 = box
    left, top, right, bottom = window
    clipped = [max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return [clipped[0]-left, clipped[1]-top, clipped[2]-left, clipped[3]-top]


def crop_record(record, window):
    result = deepcopy(record)
    left, top, right, bottom = window
    result.image_id = digest([record.image_id, list(window)])
    result.width, result.height = right-left, bottom-top
    result.metadata["transform"] = {
        "parent_image_id": record.image_id, "parent_path": record.path, "parent_sha256": record.sha256,
        "window": list(window), "scale": [1.0, 1.0], "padding": [0, 0],
    }
    result.annotations = []
    for original in record.annotations:
        if original.geometry.type != "bbox":
            raise ValueError("Tiling v1 aceita apenas anotações BBox")
        box = clip_box(original.geometry.coordinates, window)
        if box is not None:
            annotation = deepcopy(original)
            annotation.annotation_id = digest([original.annotation_id, result.image_id])
            annotation.geometry.coordinates = box
            annotation.source["parent_annotation_id"] = original.annotation_id
            full_area = ((original.geometry.coordinates[2]-original.geometry.coordinates[0]) *
                         (original.geometry.coordinates[3]-original.geometry.coordinates[1]))
            annotation.flags["retained_fraction"] = (box[2]-box[0])*(box[3]-box[1])/full_area
            result.annotations.append(annotation)
    return result


def reproject_box(box, transform):
    sx, sy = transform["scale"]
    px, py = transform["padding"]
    left, top, _, _ = transform["window"]
    if sx <= 0 or sy <= 0:
        raise ValueError("Escala inválida")
    return [(box[0]-px)/sx+left, (box[1]-py)/sy+top,
            (box[2]-px)/sx+left, (box[3]-py)/sy+top]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections: list[Detection], threshold: float) -> list[Detection]:
    if not 0 <= threshold <= 1:
        raise ValueError("Limiar NMS inválido")
    result = []
    for item in sorted(detections, key=lambda d: (-d.score, d.class_id, d.bbox_xyxy)):
        if not any(item.class_id == other.class_id and iou(item.bbox_xyxy, other.bbox_xyxy) > threshold for other in result):
            result.append(item)
    return result


def consolidate(predictions, records, threshold: float):
    predictions = list(predictions)
    index = {r.image_id: r for r in records}
    if len({p.image_id for p in predictions}) != len(predictions):
        raise ValueError("Resultados duplicados por amostra")
    if {p.image_id for p in predictions} != set(index):
        raise ValueError("Consolidação exige resultados de todas as amostras")
    if any(p.status != "processed" for p in predictions):
        raise ValueError("Consolidação contém predição com falha")
    grouped = {}
    for prediction in predictions:
        r = index[prediction.image_id]
        transform = r.metadata.get("transform")
        image_id = transform["parent_image_id"] if transform else r.image_id
        detections = grouped.setdefault(image_id, [])
        for d in prediction.detections:
            detections.append(Detection(d.class_id, reproject_box(d.bbox_xyxy, transform) if transform else list(d.bbox_xyxy), d.score))
    from ntv_ai.domain import PredictionRecord
    return [PredictionRecord(key, nms(value, threshold)) for key, value in sorted(grouped.items())]
