from dataclasses import asdict
from pathlib import Path

from ntv_ai.evaluation.bbox import evaluate
from ntv_ai.experiments.cross import evaluation_data, load_dataset, check_leakage, pair_ready
from ntv_ai.models.common import artifact, verify_framework_images
from ntv_ai.models.inference import predict_samples
from ntv_ai.storage import atomic_json


def work_cross(model, config, request, output):
    from ntv_ai.pipelines.experiment_worker import merged_predictions, save_predictions
    target = request['target']
    decision = pair_ready(config['dataset'], target, request['split'])
    if decision['state'] != 'pending':
        raise ValueError(str(decision))
    classes = decision['classes']
    manifest, originals, samples = evaluation_data(target, classes, request['split'])
    _, source_records = load_dataset(config['dataset'])
    check_leakage(source_records, originals)
    verify_framework_images(samples, manifest['sources'])
    # The predictor always uses the source head and source class order.
    predictor = model.load_predictor(Path(request['checkpoint']), config)
    predictions, timings = predict_samples(predictor, samples, manifest['sources'])
    predictions = merged_predictions(predictions, samples, config['protocol'])
    save_predictions(output/'predictions-source-classes.jsonl', predictions)
    for record in predictions:
        record.detections = [d for d in record.detections if d.class_id in classes]
    path = output/'predictions.jsonl'
    save_predictions(path, predictions)
    opts = {'split':request['split'],'max_detections':config['protocol']['max_detections']}
    metrics = evaluate(manifest, originals, path, opts)
    if metrics != evaluate(manifest, originals, path, opts):
        raise RuntimeError('Reavaliação cruzada divergente')
    atomic_json(output/'metrics.json', metrics)
    atomic_json(output/'timings.json', timings)
    return {'purpose':config['purpose'], 'source_run':request['source_run'],
            'source_version':config['dataset']['version'], 'target_version':target['version'],
            'classes':classes,'metrics':metrics,'predictions':asdict(artifact(path)),
            'excluded_unverified_classes':decision.get('excluded_unverified_classes', []),
            'reevaluation_equal':True}
