"""Explicit Qwen output policy and operating-point metrics shared by both VLMs."""
import json
import math
import re
from collections import defaultdict

from ntv_ai.domain.records import Detection
from ntv_ai.transforms.geometry import iou


def parse_qwen(text, token_spans, token_logprobs, classes, size):
    """No JSON repair or model-supplied confidence. Invalid objects remain misses."""
    rejected = []
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique_keys)
        if not isinstance(value, dict) or set(value) != {'detections'} or not isinstance(value['detections'], list):
            raise ValueError('Expected detections array')
        if len(value['detections']) > 100:
            raise ValueError('More than 100 detections')
    except (ValueError, TypeError) as exc:
        return [], [{'reason': 'invalid_json', 'detail': str(exc)}]
    if len(token_spans) != len(token_logprobs):
        raise ValueError('Token/log probability alignment failure')
    decoder, cursor, found = json.JSONDecoder(), text.index('[') + 1, []
    for index, obj in enumerate(value['detections']):
        cursor += len(text[cursor:]) - len(text[cursor:].lstrip(' \t\r\n,'))
        decoded, end = decoder.raw_decode(text, cursor)
        if decoded != obj:
            raise ValueError('JSON object alignment failure')
        fragment, start = text[cursor:end], cursor
        cursor = end
        try:
            if not isinstance(obj, dict) or set(obj) != {'class_id', 'bbox_xyxy'}:
                raise ValueError('Unexpected object keys')
            if obj['class_id'] not in classes:
                raise ValueError('Unknown class')
            box = obj['bbox_xyxy']
            if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
                raise ValueError('Invalid coordinates')
            if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                raise ValueError('Out of bounds or degenerate box')
            label_match = re.search(r'"class_id"\s*:\s*"([^"\\]*)"', fragment)
            coord_match = re.search(r'"bbox_xyxy"\s*:\s*\[([^\]]*)\]', fragment)
            if not label_match or not coord_match:
                raise ValueError('Unsupported escaped class or coordinates')
            spans = [(start + label_match.start(1), start + label_match.end(1))]
            spans += [(start + coord_match.start(1) + m.start(), start + coord_match.start(1) + m.end())
                      for m in re.finditer(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', coord_match.group(1))]
            selected = [p for (a, b), p in zip(token_spans, token_logprobs)
                        if b > a and any(a < y and b > x for x, y in spans)]
            if not selected or any(not math.isfinite(v) or v > 1e-5 for v in selected):
                raise ValueError('Invalid token probabilities')
            score = math.exp(sum(selected) / len(selected))
            scaled = [v * size[i % 2] / 1000 for i, v in enumerate(box)]
            if score >= .001:
                found.append(Detection(obj['class_id'], scaled, min(1., score)))
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({'index': index, 'reason': 'invalid_object', 'detail': str(exc)})
    return sorted(found, key=lambda d: (-d.score, d.class_id, d.bbox_xyxy))[:100], rejected


def operating_metrics(records, predictions, classes, threshold):
    lookup = {p['image_id']: p for p in predictions}
    if set(lookup) != {r.image_id for r in records} or len(lookup) != len(predictions):
        raise ValueError('Operating metrics require exactly one result per image')
    counts = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in classes}
    for r in records:
        truth = defaultdict(list)
        for a in r.annotations:
            if a.mapping_status == 'mapped' and a.label in classes:
                truth[a.label].append(a.geometry.coordinates)
        used = defaultdict(set)
        considered = defaultdict(int)
        for d in sorted(lookup[r.image_id]['detections'], key=lambda d: -d['score']):
            c = d['class_id']
            if c not in counts or d['score'] < threshold:
                continue
            if considered[c] >= 100:
                continue
            considered[c] += 1
            eligible = [(iou(d['bbox_xyxy'], box), i) for i, box in enumerate(truth[c]) if i not in used[c]]
            overlap, index = max(eligible, default=(0., -1))
            if overlap >= .5:
                used[c].add(index)
                counts[c]['tp'] += 1
            else:
                counts[c]['fp'] += 1
        for c in classes:
            counts[c]['fn'] += len(truth[c]) - len(used[c])
    def metrics(v):
        tp, fp, fn = v['tp'], v['fp'], v['fn']
        return {**v, 'precision': tp/(tp+fp) if tp+fp else 0.,
                'recall': tp/(tp+fn) if tp+fn else 0., 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    total = {k: sum(v[k] for v in counts.values()) for k in ('tp', 'fp', 'fn')}
    return {'iou': .5, 'threshold': threshold, 'micro': metrics(total),
            'per_class': {c: metrics(v) for c, v in counts.items()}}


def choose_threshold(records, predictions, classes):
    grid = [.001, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
    rows = [operating_metrics(records, predictions, classes, v) for v in grid]
    return max(rows, key=lambda r: (r['micro']['f1'], -r['threshold']))
