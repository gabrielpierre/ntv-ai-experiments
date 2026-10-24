"""One resumable inference cell; infrastructure failures never become empty predictions."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

from ntv_ai.domain.records import Detection, PredictionRecord
from ntv_ai.evaluation.bbox import evaluate
from ntv_ai.experiments.cross import load_dataset, evaluation_data
from ntv_ai.experiments.workshop_metrics import choose_threshold, operating_metrics
from ntv_ai.experiments.workshop_prompts import CLASSES, bank
from ntv_ai.models.runtime import gpu_slot
from ntv_ai.pipelines.experiment_worker import merged_predictions, save_predictions
from ntv_ai.storage import atomic_json, digest, file_hash, inside, load_json


def work(request):
    import torch
    from PIL import Image
    from ntv_ai.models.backends.workshop_vlm import VlmPredictor
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    identity = digest({k: v for k, v in request.items() if k not in ('output', 'deadline')} | {'prompts': bank()})
    identity_path = output / 'identity.json'
    if identity_path.exists():
        if load_json(identity_path)['identity'] != identity:
            raise ValueError('Resumed inference identity changed')
    else:
        atomic_json(identity_path, {'identity': identity, 'request': request})
    if (output / 'result.json').exists():
        return load_json(output / 'result.json')
    if request.get('smoke'):
        manifest, rows = load_dataset(request['dataset'])
        ids = set(manifest['view']['image_ids'])
        samples = sorted((r for r in rows if r.split == 'train' and r.image_id in ids), key=lambda r:r.image_id)[:2]
        originals = []
    else:
        manifest, originals, samples = evaluation_data(request['dataset'], CLASSES, request['split'])
    samples = sorted(samples, key=lambda r: r.image_id)
    predictions, invalid, timings = [], [], []
    started = time.monotonic()
    with gpu_slot(0):
        torch.set_num_threads(8)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        predictor = VlmPredictor(request['model'], request['asset']['path'], request['resolution'], request['prompt_id'])
        for record in samples:
            sample_path = output / 'samples' / f'{record.image_id}.json'
            if sample_path.exists():
                saved = load_json(sample_path)
                if saved['identity'] != identity or saved['image_sha256'] != record.sha256:
                    raise ValueError('Cached prediction changed')
            else:
                path = inside(Path(manifest['sources'][record.dataset_id]['root']), record.path)
                if file_hash(path) != record.sha256:
                    raise ValueError('Source image changed')
                with Image.open(path) as image:
                    if image.size != (record.width, record.height):
                        raise ValueError('Source dimensions changed')
                    items, elapsed, raw = predictor(image.convert('RGB'))
                saved = {'identity': identity, 'image_sha256': record.sha256,
                         'prediction': asdict(PredictionRecord(record.image_id, items)),
                         'seconds': elapsed, 'raw': raw}
                atomic_json(sample_path, saved)
            p = saved['prediction']
            predictions.append(PredictionRecord(p['image_id'], [Detection(**d) for d in p['detections']]))
            if saved['raw']['invalid_outputs']:
                invalid.append({'image_id': record.image_id, 'errors': saved['raw']['invalid_outputs']})
            timings.append({'image_id': record.image_id, 'seconds': saved['seconds']})
            atomic_json(output / 'progress.json', {'completed': len(predictions), 'total': len(samples),
                        'last_image': record.image_id, 'identity': identity}, replace=True)
        peak = torch.cuda.max_memory_allocated()
        del predictor
        torch.cuda.empty_cache()
    result = {'identity': identity, 'model': request['model'], 'protocol': request['protocol'],
              'prompt_id': request['prompt_id'], 'split': request.get('split'),
              'invalid_samples': invalid, 'timings': timings, 'peak_allocated_bytes': peak,
              'elapsed_seconds': time.monotonic() - started,
              'score_policy': 'native_similarity' if request['model'] == 'grounding_dino'
                  else 'uncalibrated_geometric_mean_token_probability',
              'invalid_policy': 'Invalid model output is scored as missed detections, never omitted; infrastructure errors abort.',
              'smoke_only': bool(request.get('smoke'))}
    if not request.get('smoke'):
        merged = merged_predictions(predictions, samples, {'tile_iou': .5})
        path = output / 'predictions.jsonl'
        save_predictions(path, merged)
        config = {'split': request['split'], 'max_detections': 100}
        metrics = evaluate(manifest, originals, path, config)
        if metrics != evaluate(manifest, originals, path, config):
            raise ValueError('Metric replay differs')
        rows = [asdict(p) for p in merged]
        operating = (choose_threshold(originals, rows, CLASSES) if request['split'] == 'val'
                     else operating_metrics(originals, rows, CLASSES, request['threshold']))
        subset_manifest = deepcopy(manifest)
        subset_manifest['view']['classes'] = ['isolador', 'espacador', 'amortecedor']
        subset = deepcopy(merged)
        for p in subset:
            p.detections = [d for d in p.detections if d.class_id in subset_manifest['view']['classes']]
        subset_path = output / 'predictions-three-classes.jsonl'
        save_predictions(subset_path, subset)
        result.update(metrics=metrics, operating=operating,
                      subset_metrics=evaluate(subset_manifest, originals, subset_path, config),
                      subset_operating=operating_metrics(originals, [asdict(p) for p in subset],
                          subset_manifest['view']['classes'], operating['threshold']),
                      predictions={'path': str(path), 'sha256': file_hash(path)}, reevaluation_equal=True)
    atomic_json(output / 'result.json', result)
    return result


def main():
    import traceback
    args = argparse.ArgumentParser()
    args.add_argument('request', type=Path)
    request = load_json(args.parse_args().request)
    try:
        work(request)
    except BaseException as exc:
        atomic_json(Path(request['output']) / 'failure.json',
                    {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}, replace=True)
        raise


if __name__ == '__main__':
    main()
