import math
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from ntv_ai.data.export import export_views
from ntv_ai.data.views import bbox_errors
from ntv_ai.domain.records import Annotation, ArtifactReference, Detection, Geometry, PredictionRecord, PreparedDataset, TrainingResult
from ntv_ai.storage import atomic_json, file_hash, inside, read_snapshot, safe_output
from ntv_ai.transforms.geometry import consolidate
from ntv_ai.models.common import artifact, checked_artifact, prepared_dataset, verify_framework_images


def class_names(names) -> list[str]:
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("IDs de classes não contíguos no checkpoint")
        return [names[i] for i in range(len(names))]
    return list(names)


def check_classes(names, classes):
    if class_names(names) != list(classes):
        raise ValueError("Classes do checkpoint incompatíveis com a view")


def load_model(checkpoint: Path):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="detect")
    architecture = model.model.yaml
    if model.task != "detect" or architecture.get("scale") != "l" or "yolo26" not in Path(architecture.get("yaml_file", "")).stem:
        raise ValueError("Checkpoint não pertence à arquitetura YOLO26-L")
    if getattr(model.model.model[-1], "one2one_cv2", None) is None:
        raise ValueError("Checkpoint sem cabeça end-to-end")
    return model


def disable_oom_retry(trainer):
    trainer._oom_retries = 3


@contextmanager
def cuda_runtime(output: Path, device: int):
    if type(device) is not int or device < 0:
        raise ValueError("device deve ser o índice de uma GPU CUDA")
    import torch

    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError("GPU CUDA solicitada indisponível; execução interrompida")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    probe = torch.ones((2, 2), device=f"cuda:{device}")
    if (probe @ probe).sum().item() != 8:
        raise RuntimeError("Operação CUDA de verificação falhou")
    properties = torch.cuda.get_device_properties(device)
    info = {"device": f"cuda:{device}", "gpu": properties.name, "vram": properties.total_memory,
            "cuda": torch.version.cuda, "torch": torch.__version__}
    for name, folder in (("YOLO_CONFIG_DIR", "settings"), ("MPLCONFIGDIR", "matplotlib"),
                         ("XDG_CACHE_HOME", "cache")):
        (output / folder).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(output / folder)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import settings
    from ultralytics.utils import LOGGER
    from importlib.metadata import version

    if version("ultralytics") != "8.4.155":
        raise RuntimeError("Este backend requer Ultralytics 8.4.155")
    settings.file = output / "settings/Ultralytics/settings.json"
    settings.file.parent.mkdir(parents=True, exist_ok=True)
    settings.update({name: False for name in ("sync", "clearml", "comet", "dvc", "hub", "mlflow",
                                             "neptune", "raytune", "tensorboard", "wandb") if name in settings})
    settings.update({key: str(output / key) for key in ("datasets_dir", "weights_dir", "runs_dir")})
    handler = logging.FileHandler(output / "native.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        yield info
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
        info["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        info["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        atomic_json(output / "runtime.json", info)


TRAIN_OPTIONS = {"epochs", "imgsz", "batch", "device", "seed", "workers", "optimizer", "lr0", "lrf",
                 "nbs", "warmup_epochs", "mosaic", "close_mosaic", "amp", "compile", "momentum",
                 "weight_decay", "patience", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                 "shear", "perspective", "flipud", "fliplr", "mixup", "cutmix", "copy_paste", "deterministic"}
PREDICT_OPTIONS = {"imgsz", "device", "conf", "max_det", "nms", "tile_iou"}


class YOLO26LBackend:
    def train(self, dataset: PreparedDataset, config: dict, output: Path) -> TrainingResult:
        manifest, records = prepared_dataset(dataset)
        if not all(any(r.split == split for r in records) for split in ("train", "val")):
            raise ValueError("Treinamento requer imagens de treino e validação")
        verify_framework_images(records, manifest["sources"])
        checkpoint = checked_artifact(ArtifactReference(**config["checkpoint"]))
        options = dict(config["training"])
        if options.keys() - TRAIN_OPTIONS:
            raise ValueError(f"Opções de treino não suportadas: {sorted(options.keys() - TRAIN_OPTIONS)}")
        if type(options.get("batch")) is not int or options["batch"] < 1:
            raise ValueError("batch deve ser inteiro positivo, sem seleção automática")
        protected = [Path(s["root"]) for s in manifest["sources"].values()]
        output = safe_output(output, [*protected, Path(dataset.manifest.path).parent, checkpoint.parent])
        output.mkdir(parents=True, exist_ok=False)
        export_views(records, manifest["sources"], list(dataset.classes), output / "export", ["yolo"])
        options.update(data=str(output / "export/yolo/data.yaml"), project=str(output), name="native",
                       exist_ok=False, cache=False, save=True, val=True, plots=False, resume=False,
                       save_json=False, nms=False)
        atomic_json(output / "requested_args.json", options)
        with cuda_runtime(output, options["device"]) as runtime:
            model = load_model(checkpoint)
            model.add_callback("on_train_epoch_start", disable_oom_retry)
            model.train(**options)
            if model.trainer.device.type != "cuda":
                raise RuntimeError("Treinador não utilizou CUDA")
            check_classes(model.names, dataset.classes)
            atomic_json(output / "effective_args.json", vars(model.trainer.args))
            metrics = {k: float(v) for k, v in model.metrics.results_dict.items()}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("Métricas nativas não finitas")
            checkpoints = {key: artifact(Path(getattr(model.trainer, key))) for key in ("best", "last")}
            result = TrainingResult(checkpoints, metrics, {
                "model": "yolo26l", "classes": list(dataset.classes), "dataset_version": manifest["version"],
                "view_id": dataset.view_id, "initial_checkpoint": config["checkpoint"], "runtime": runtime,
                "selection": "best.pt: fitness de validação do treinador nativo",
            })
        atomic_json(output / "training.json", asdict(result))
        return result

    def predict(self, checkpoint: ArtifactReference, images, config: dict):
        path = checked_artifact(checkpoint)
        records = list(images)
        classes = config["classes"]
        options = dict(config["inference"])
        if options.keys() - PREDICT_OPTIONS or options.get("nms") is not False:
            raise ValueError("Inferência YOLO26-L requer opções conhecidas e nms: false")
        tile_iou = options.pop("tile_iou", 0.5)
        if not 0 < tile_iou <= 1 or not 0 <= options["conf"] <= 1 or options["max_det"] < 1:
            raise ValueError("Limiares de inferência inválidos")
        if not records or len({r.image_id for r in records}) != len(records):
            raise ValueError("Inferência requer imagens únicas e não vazias")
        sources = config["sources"]
        protected = [Path(s["root"]) for s in sources.values()]
        output = safe_output(Path(config["output"]), [*protected, path.parent])
        output.mkdir(parents=True, exist_ok=False)
        predictions = []
        with cuda_runtime(output, options["device"]):
            model = load_model(path)
            check_classes(model.names, classes)
            atomic_json(output / "requested_args.json", options)
            for record in records:
                source = inside(Path(sources[record.dataset_id]["root"]), record.path)
                if file_hash(source) != record.sha256:
                    raise ValueError(f"Imagem alterada: {record.image_id}")
                with Image.open(source) as image:
                    if image.size != (record.width, record.height):
                        raise ValueError("Dimensões divergentes na inferência")
                    results = model.predict(image.convert("RGB"), **options, save=False, verbose=False,
                                            half=False, compile=False, augment=False)
                if len(results) != 1 or tuple(results[0].orig_shape) != (record.height, record.width):
                    raise ValueError("Resposta do modelo incompatível com a imagem")
                detections = normalize_boxes(results[0].boxes, classes, record)
                predictions.append(PredictionRecord(record.image_id, detections))
            if model.predictor.device.type != "cuda":
                raise RuntimeError("Inferência não utilizou CUDA")
            atomic_json(output / "effective_args.json", vars(model.predictor.args))
        if any("transform" in r.metadata for r in records):
            predictions = consolidate(predictions, records, tile_iou)
        return predictions


def normalize_boxes(boxes, classes, record):
    detections = []
    for coords, confidence, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                          boxes.cls.cpu().tolist(), strict=True):
        if not math.isfinite(class_id) or int(class_id) != class_id or not 0 <= class_id < len(classes):
            raise ValueError("Classe inválida retornada pelo modelo")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confiança inválida retornada pelo modelo")
        annotation = Annotation("prediction", classes[int(class_id)], Geometry("bbox", coords), {})
        if bbox_errors(annotation, record):
            raise ValueError("Caixa inválida retornada pelo modelo")
        detections.append(Detection(classes[int(class_id)], coords, confidence))
    return detections
