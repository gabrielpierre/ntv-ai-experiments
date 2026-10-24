import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from ntv_ai.domain.records import ArtifactReference, PreparedDataset
from ntv_ai.evaluation.bbox import evaluate
from ntv_ai.experiments.selection import CheckpointSelection
from ntv_ai.models.common import artifact, prepared_dataset, verify_framework_images
from ntv_ai.models.inference import predict_samples
from ntv_ai.models.registry import backend
from ntv_ai.models.runtime import gpu_slot, runtime, verify_implementation
from ntv_ai.storage import atomic_json, load_json, read_snapshot
from ntv_ai.transforms.geometry import consolidate


def save_predictions(path, predictions):
    path.write_text("".join(json.dumps(asdict(p), ensure_ascii=False, allow_nan=False) + "\n"
                            for p in sorted(predictions, key=lambda p: p.image_id)))


def merged_predictions(predictions, records, protocol):
    if any(r.metadata.get("transform") for r in records):
        return consolidate(predictions, records, protocol["tile_iou"])
    return predictions


class Observer:
    def __init__(self, model, config, manifest, records, output, parent=None, profile=False):
        self.model, self.config, self.manifest, self.records, self.output = model, config, manifest, records, output
        self.selection = CheckpointSelection(output / "selection", parent)
        self.profile = profile
        self.resume = None
        self.epoch_index = None
        self.control = None
        self.control_ref = None
        if config['training'].get('control'):
            from ntv_ai.experiments.control import TrainingControl
            state = None
            if config.get('resume'):
                state = load_json(Path(config['resume']['control']['path']))['state']
            self.control = TrainingControl(config['training']['control'], state)

    def bind_optimizer(self, optimizer, steps_per_epoch):
        if self.control:
            self.control.bind(optimizer, steps_per_epoch)
            atomic_json(self.output/'optimizer.json', {
                'type':type(optimizer).__name__, 'controller':self.control.config,
                'steps_per_epoch':steps_per_epoch, 'base_lrs':self.control.state['base_lrs'],
                'groups':[{k:v for k,v in group.items() if k != 'params'} for group in optimizer.param_groups],
                'native_scheduler':'constant; common controller applies rates before every optimizer update'})

    def epoch(self, epoch, checkpoint, resume):
        import torch
        from ntv_ai.models.runtime import rng_state, restore_rng
        self.resume, self.epoch_index = artifact(resume), epoch
        if self.profile:
            return
        state = rng_state()
        try:
            records = [r for r in self.records if r.split == "val"]
            predictor = self.model.load_predictor(checkpoint, self.config)
            predictions, _ = predict_samples(predictor, records, self.manifest["sources"])
            del predictor
            torch.cuda.empty_cache()
            predictions = merged_predictions(predictions, records, self.config["protocol"])
            path = self.output / f"val-{epoch:04d}.jsonl"
            save_predictions(path, predictions)
            _, originals = read_snapshot(Path(self.config["dataset"]["manifest"]["path"]))
            metrics = evaluate(self.manifest, originals, path, {
                "split": "val", "max_detections": self.config["protocol"]["max_detections"]})
            atomic_json(self.output / f"val-{epoch:04d}.json", metrics)
            self.selection.consider(epoch, metrics["mAP50_95"], checkpoint)
            stopped = self.control.observe(epoch, metrics['mAP50_95']) if self.control else False
            if self.control:
                control_path = resume.with_suffix(resume.suffix + '.control.json')
                atomic_json(control_path, {'state': self.control.state, 'config': self.control.config,
                                          'resume_sha256': self.resume.sha256})
                self.control_ref = artifact(control_path)
                with (self.output/'control-history.jsonl').open('a') as history:
                    history.write(json.dumps({'epoch':epoch, 'mAP50_95':metrics['mAP50_95'],
                                              **self.control.state}) + '\n')
            atomic_json(self.output / "progress.json", {"resume": asdict(self.resume), "last_epoch": epoch,
                'control': asdict(self.control_ref) if self.control_ref else None}, replace=True)
            if self.control:
                from ntv_ai.experiments.control import retain_checkpoints
                retain_checkpoints(resume, self.control.config['keep_resume'])
            return stopped
        finally:
            restore_rng(state)


def work(request):
    config = request["config"]
    output = Path(request["output"])
    dataset = PreparedDataset(ArtifactReference(**config["dataset"]["manifest"]),
                              config["dataset"]["view_id"], tuple(config["dataset"]["classes"]))
    manifest, records = prepared_dataset(dataset, config["dataset"].get("source_roots"))
    verify_framework_images(records, manifest["sources"])
    config["sources"] = manifest["sources"]
    config["classes"] = list(dataset.classes)
    verify_implementation(config)
    with gpu_slot(config["runtime"]["device"], request.get("lease_fd")), runtime(config, output) as info:
        model = backend(config["model"]["backend"])
        if request['operation'] == 'cross_predict':
            from ntv_ai.pipelines.cross_worker import work_cross
            return work_cross(model, config, request, output)
        if request["operation"] in ("train", "smoke", "profile"):
            observer = Observer(model, config, manifest, records, output,
                                config.get("resume", {}).get("selection"), request["operation"] == "profile")
            result = model.train(dataset, config, output / "backend", observer)
            if result.metadata.get("optimizer_steps", 0) < 1:
                raise RuntimeError("O treinador não confirmou nenhuma atualização do otimizador")
            info["optimizer_steps"] = result.metadata["optimizer_steps"]
            if observer.resume is None:
                raise RuntimeError("Checkpoint completo de retomada não foi produzido")
            summary = {"training": asdict(result), "resume": asdict(observer.resume),
                       "last_epoch": observer.epoch_index, "purpose": config["purpose"]}
            if observer.control:
                summary.update(control=asdict(observer.control_ref) if observer.control_ref else None,
                               stop_reason='early_stopping' if observer.control.state['stopped'] else 'epoch_limit',
                               control_state=observer.control.state)
            if request["operation"] == "profile":
                return {**summary, "feasibility": "passed", "validation_included": True}
            if observer.selection.best is None:
                raise RuntimeError("Seleção comum de checkpoint não foi executada")
            summary["best"] = observer.selection.best
            checkpoint = Path(summary["best"]["checkpoint"]["path"])
        else:
            summary = {"purpose": config["purpose"]}
            checkpoint = Path(request["checkpoint"])
        if request["operation"] in ("smoke", "predict", "measure"):
            predictor = model.load_predictor(checkpoint, config)
            selected = [r for r in records if r.split == request.get("split", "val")]
            if not selected:
                raise ValueError("Split vazio")
            if request["operation"] == "measure":
                summary["latency"] = measure(predictor, selected, manifest["sources"], config["protocol"])
            predictions, timings = predict_samples(predictor, selected, manifest["sources"])
            predictions = merged_predictions(predictions, selected, config["protocol"])
            path = output / "predictions.jsonl"
            save_predictions(path, predictions)
            _, originals = read_snapshot(Path(dataset.manifest.path))
            opts = {"split": request.get("split", "val"), "max_detections": config["protocol"]["max_detections"]}
            metrics = evaluate(manifest, originals, path, opts)
            repeated = evaluate(manifest, originals, path, opts)
            if metrics != repeated:
                raise RuntimeError("Reavaliação das mesmas predições divergiu")
            atomic_json(output / "metrics.json", metrics)
            atomic_json(output / "timings.json", timings)
            summary.update(predictions=asdict(artifact(path)), metrics=metrics, reevaluation_equal=True)
        return summary


def measure(predictor, records, sources, protocol):
    import statistics
    import time
    import torch
    from PIL import Image
    from ntv_ai.domain.records import PredictionRecord
    from ntv_ai.storage import inside, file_hash
    groups = {}
    for r in records:
        parent = r.metadata.get("transform", {}).get("parent_image_id", r.image_id)
        groups.setdefault(parent, []).append(r)
    warmup, repeats = protocol.get("warmup", 5), protocol.get("repeats", 10)
    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("Aquecimento e repetições devem ser positivos")

    def whole_image(samples):
        first = samples[0]
        transform = first.metadata.get("transform")
        dataset_id = first.metadata["parent_dataset_id"] if transform else first.dataset_id
        relative = transform["parent_path"] if transform else first.path
        source = inside(Path(sources[dataset_id]["root"]), relative)
        expected = transform["parent_sha256"] if transform else first.sha256
        if file_hash(source) != expected:
            raise ValueError("Imagem original alterada durante medição")
        predictions, model_seconds = [], 0.0
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            for record in samples:
                transform = record.metadata.get("transform")
                if transform:
                    if transform["scale"] != [1.0, 1.0] or transform["padding"] != [0, 0]:
                        raise ValueError("Medição da imagem completa v1 requer tiles sem resize/padding externo")
                    sample = image.crop(transform["window"])
                else:
                    sample = image
                boxes, seconds = predictor(sample)
                if not 0 < seconds < float("inf"):
                    raise ValueError("Backend não registrou tempo válido do modelo")
                model_seconds += seconds
                predictions.append(PredictionRecord(record.image_id, boxes))
        merged_predictions(predictions, samples, protocol)
        return model_seconds

    for _ in range(warmup):
        whole_image(list(groups.values())[0])
    measurements = []
    for _ in range(repeats):
        for parent, samples in groups.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            model_seconds = whole_image(samples)
            torch.cuda.synchronize()
            measurements.append({"image_id": parent, "samples": len(samples),
                                 "model_seconds": model_seconds,
                                 "image_seconds": time.perf_counter() - start})
    return {"batch": 1, "warmup": warmup, "repeats": repeats, "cuda_synchronized": True,
            "includes": "original image hash/read/decode, tile extraction, preprocessing, model, postprocessing and consolidation; excludes model loading",
            "mean_model_seconds": statistics.mean(m["model_seconds"] for m in measurements),
            "mean_image_seconds": statistics.mean(m["image_seconds"] for m in measurements),
            "measurements": measurements}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = load_json(args.request)
    output = Path(request["output"])
    try:
        result = work(request)
        atomic_json(output / "result.json", result)
        return 0
    except BaseException as exc:
        is_oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        atomic_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                    "oom": is_oom, "recipe_changed": False, "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
