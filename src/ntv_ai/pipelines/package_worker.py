import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from ntv_ai.domain import PredictionRecord
from ntv_ai.models.inference import predict_image
from ntv_ai.models.package import ModelPackage
from ntv_ai.models.registry import backend
from ntv_ai.models.runtime import gpu_slot, runtime, verify_implementation
from ntv_ai.pipelines.model_packages import check_external_implementation
from ntv_ai.storage import atomic_json, load_json, file_hash


def work(request):
    package = ModelPackage(request["package"])
    row = package.select(request["model"], request["protocol"], request["seed"])
    ref = package.reference(row, "best_inference")
    checkpoint = package.file(ref["package_path"])
    config, output = request["config"], Path(request["output"])
    check_external_implementation(config)
    verify_implementation(config)
    import torch
    timings = []
    with gpu_slot(config["runtime"]["device"]), runtime(config, output):
        predictor = backend(row["backend"]).load_predictor(checkpoint, config)
        predictions = output / "predictions.jsonl"
        with predictions.open("x") as stream:
            for record in request["images"]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                with Image.open(record["path"]) as image:
                    items, timing = predict_image(predictor, image.convert("RGB"), request["tiling"], config["protocol"]["tile_iou"])
                torch.cuda.synchronize()
                timings.append({"image_id": record["image_id"], **timing, "image_seconds": time.perf_counter() - start})
                prediction = PredictionRecord(record["image_id"], items)
                stream.write(json.dumps(asdict(prediction), ensure_ascii=False, allow_nan=False) + "\n")
    result = {"images": len(request["images"]), "classes": config["classes"], "source_run": row["run_id"],
              "checkpoint": str(checkpoint), "protocol": request["protocol"],
              "checkpoint_sha256": file_hash(checkpoint),
              "predictions": {"path": str(predictions), "sha256": file_hash(predictions)},
              "timings": timings, "timing_note": "diagnostic only; no warmup, not a latency benchmark",
              "implementation_check": "installed package versions"}
    if request.get('evaluation_target'):
        from ntv_ai.experiments.cross import evaluation_data
        from ntv_ai.evaluation.bbox import evaluate
        manifest, originals, _ = evaluation_data(request['evaluation_target'], request['evaluation_classes'], request['split'])
        filtered = output / 'predictions-evaluated.jsonl'
        records = [json.loads(line) for line in predictions.read_text().splitlines()]
        for record in records:
            record['detections'] = [d for d in record['detections'] if d['class_id'] in request['evaluation_classes']]
        filtered.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
        opts = {'split': request['split'], 'max_detections': config['protocol']['max_detections']}
        metrics = evaluate(manifest, originals, filtered, opts)
        if metrics != evaluate(manifest, originals, filtered, opts):
            raise RuntimeError('Package reevaluation differs')
        atomic_json(output / 'metrics.json', metrics)
        result.update(metrics=metrics, reevaluation_equal=True, evaluated_classes=request['evaluation_classes'])
    atomic_json(output / "result.json", result)


if __name__ == "__main__":
    request = load_json(Path(sys.argv[1]))
    try:
        work(request)
    except Exception as exc:
        failure = {"message": str(exc), "type": type(exc).__name__, "oom": "out of memory" in str(exc).lower(),
                   "recipe_changed": False, "traceback": traceback.format_exc()}
        atomic_json(Path(request["output"]) / "failure.json", failure)
        raise
