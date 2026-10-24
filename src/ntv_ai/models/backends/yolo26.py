from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from ntv_ai.domain.records import TrainingResult
from ntv_ai.models.common import artifact, checked_artifact, prepared_dataset
from ntv_ai.models.exports import export_dataset
from ntv_ai.models.inference import detections, predict_samples
from ntv_ai.models.runtime import OptimizerSteps, restore_rng, rng_state
from ntv_ai.storage import atomic_json


NATIVE_OPTIONS = {"optimizer", "lr0", "lrf", "warmup_epochs", "mosaic", "close_mosaic", "momentum",
                  "weight_decay", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                  "perspective", "flipud", "fliplr", "mixup", "cutmix", "copy_paste", "loss_horizon"}


def options(config, data, output):
    training, protocol = config["training"], config["protocol"]
    native = dict(training.get("native", {}))
    if native.keys() - NATIVE_OPTIONS:
        raise ValueError(f"Opções YOLO26 não suportadas: {sorted(native.keys() - NATIVE_OPTIONS)}")
    native.pop('loss_horizon', None)
    if training.get('control'):
        native.update(warmup_epochs=0, lrf=1.0)
    return {**native, "epochs": training["epochs"], "batch": training["batch"], "nbs": training["batch"],
            "imgsz": protocol["resolution"], "device": config["runtime"]["device"], "seed": training["seed"],
            "workers": training["workers"], "amp": training["amp"], "data": str(data),
            "project": str(output), "name": "native", "exist_ok": False, "cache": False,
            "save": True, "val": True, "plots": False, "patience": 0, "compile": False,
            "deterministic": True, "nms": False, "save_json": False, "save_period": -1}


def load_model(path, variant):
    if version("ultralytics") != "8.4.155":
        raise RuntimeError("YOLO26 requer Ultralytics 8.4.155")
    from ultralytics import YOLO
    model = YOLO(str(path), task="detect")
    architecture = model.model.yaml
    if architecture.get("scale") != variant or "yolo26" not in Path(architecture.get("yaml_file", "")).stem:
        raise ValueError("Checkpoint não pertence à variante YOLO26 solicitada")
    if getattr(model.model.model[-1], "one2one_cv2", None) is None:
        raise ValueError("Checkpoint sem cabeça end-to-end")
    return model


class Backend:
    def train(self, dataset, config, output, observer):
        import torch
        from unittest.mock import patch
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics import settings
        from ntv_ai.models.backends.yolo26l import disable_oom_retry

        class Trainer(DetectionTrainer):
            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
                return super().get_dataloader(dataset_path, 1 if mode == "val" else batch_size, rank, mode)

            def check_resume(self, overrides):
                super().check_resume(overrides)
                if self.resume:
                    for key in ("project", "name", "epochs", "data", "save_dir"):
                        if key in overrides:
                            setattr(self.args, key, overrides[key])

            def _load_checkpoint_state(self, ckpt):
                super()._load_checkpoint_state(ckpt)
                if self.resume:
                    if "ntv_resume" not in ckpt:
                        raise ValueError("Retomada requer checkpoint completo produzido pelo núcleo")
                    state = ckpt["ntv_resume"]
                    self.model.load_state_dict(state["model"])
                    self.scheduler.load_state_dict(state["scheduler"])
                    self.ntv_rng = state["rng"]

        output.mkdir(parents=True, exist_ok=False)
        manifest, records = prepared_dataset(dataset, config["dataset"].get("source_roots"))
        data = export_dataset(records, manifest["sources"], dataset.classes, output / "export", "yolo")
        opts = options(config, data, output)
        initial = config.get("resume", {}).get("checkpoint", config["model"]["weights"])["path"]
        model = load_model(initial, config["model"]["variant"])
        for key in ("sync", "clearml", "comet", "dvc", "hub", "mlflow", "neptune", "raytune", "tensorboard", "wandb"):
            if key in settings:
                settings.update({key: False})
        steps = OptimizerSteps()
        def start(trainer):
            disable_oom_retry(trainer)
            steps.attach(trainer.optimizer)
            observer.bind_optimizer(trainer.optimizer, len(trainer.train_loader))
            if config['training'].get('control'):
                criterion = trainer.model.init_criterion()
                criterion.one2one.hyp = deepcopy(criterion.one2one.hyp)
                criterion.one2one.hyp.epochs = config['training']['native']['loss_horizon']
                criterion.updates = trainer.start_epoch
                criterion.o2m = criterion.decay(criterion.updates)
                criterion.o2o = criterion.total - criterion.o2m
                trainer.model.criterion = criterion
            if config["training"]["amp"] and not config.get("resume"):
                trainer.scaler = torch.amp.GradScaler("cuda", init_scale=config["training"].get("amp_initial_scale", 256))
            if hasattr(trainer, "ntv_rng"):
                restore_rng(trainer.ntv_rng)
            atomic_json(output / "effective_args.json", vars(trainer.args))

        def saved(trainer):
            # Preserve full precision regular weights and optimizer before Ultralytics
            # strips its native checkpoints at train_end. Inference always uses EMA.
            full = torch.load(trainer.last, map_location="cpu", weights_only=False)
            full["optimizer"] = deepcopy(trainer.optimizer.state_dict())
            full["ema"] = deepcopy(trainer.ema.ema).cpu().float()
            full["ntv_resume"] = {"model": deepcopy(trainer.model).cpu().float().state_dict(),
                                   "scheduler": trainer.scheduler.state_dict(), "rng": rng_state()}
            resume = output / f"resume-{trainer.epoch:04d}.pt"
            torch.save(full, resume)
            candidate = output / "candidate.pt"
            torch.save({"model": deepcopy(trainer.ema.ema).cpu().float(), "ema": None,
                        "train_args": vars(trainer.args), "epoch": -1}, candidate)
            if observer.epoch(trainer.epoch, candidate, resume):
                trainer.stop = True

        model.add_callback("on_train_start", start)
        model.add_callback("on_model_save", saved)
        atomic_json(output / "requested_args.json", opts)
        # Native AMP's asset-based check downloads an unrelated model. The explicit
        # recipe is tested by its own forward/backward; no downloads or fallback.
        with patch("ultralytics.engine.trainer.check_amp", return_value=True):
            model.train(trainer=Trainer, resume=initial if config.get("resume") else False, **opts)
        if config['training'].get('control'):
            for checkpoint in (output/'native/weights').glob('*.pt'):
                checkpoint.unlink()
        return TrainingResult({"resume": observer.resume}, {},
                              {"optimizer_steps": steps.count, "weight_policy": "ema", "effective_batch": opts["nbs"]})

    def load_predictor(self, checkpoint, config):
        from ntv_ai.models.backends.yolo26l import check_classes
        model = load_model(checkpoint, config["model"]["variant"])
        check_classes(model.names, config["classes"])
        protocol = config["protocol"]
        def predict(image):
            result = model.predict(image, imgsz=protocol["resolution"], device=config["runtime"]["device"],
                conf=protocol["confidence"], max_det=protocol["max_detections"], nms=False,
                half=False, compile=False, augment=False, rect=False, save=False, verbose=False)[0]
            boxes = result.boxes
            return detections(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(),
                              config["classes"], image.size, protocol["confidence"], protocol["max_detections"], clip=True), result.speed["inference"] / 1000
        return predict

    def predict(self, checkpoint, images, config):
        predictor = self.load_predictor(checked_artifact(checkpoint), config)
        return predict_samples(predictor, list(images), config["sources"])[0]
