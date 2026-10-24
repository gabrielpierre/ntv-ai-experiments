from importlib.metadata import version
from pathlib import Path

from ntv_ai.domain.records import TrainingResult
from ntv_ai.models.common import artifact, checked_artifact, prepared_dataset
from ntv_ai.models.exports import export_dataset
from ntv_ai.models.inference import detections, predict_samples
from ntv_ai.models.runtime import ForwardTimer, OptimizerSteps, configure_precision, restore_rng, rng_state
from ntv_ai.storage import atomic_json

NATIVE_OPTIONS = {"lr", "lr_encoder", "weight_decay", "warmup_epochs", "clip_max_norm"}


def foreground_outputs(outputs, classes):
    # RF-DETR 1.11 builds C+1 logits (the final slot is background), but its
    # native sigmoid top-k includes that slot. Remove it before top-k, without
    # resizing or replacing the trained classification head.
    logits = outputs['pred_logits']
    if logits.shape[-1] != len(classes) + 1:
        raise ValueError('Cabeça RF-DETR diverge das classes de treinamento')
    return {**outputs, 'pred_logits': logits[..., :-1]}


def options(config, data, output):
    training, protocol = config["training"], config["protocol"]
    native = training.get("native", {})
    if native.keys() - NATIVE_OPTIONS:
        raise ValueError(f"Opções RF-DETR não suportadas: {sorted(native.keys() - NATIVE_OPTIONS)}")
    if training.get('control'):
        native = {**native, 'warmup_epochs': 0, 'lr_scheduler': 'torch.optim.lr_scheduler.ConstantLR',
                  'lr_scheduler_kwargs': {'factor': 1.0, 'total_iters': 1}}
    return {**native, "dataset_dir": str(data), "dataset_file": "roboflow", "output_dir": str(output / "native"),
            "epochs": training["epochs"], "batch_size": training["batch"], "eval_batch_size": 1,
            "grad_accum_steps": 1, "num_workers": training["workers"], "seed": training["seed"],
            "amp_dtype": "fp16" if training["amp"] else None,
            "use_ema": protocol["weight_policy"] == "ema", "multi_scale": False, "expanded_scales": False,
            "scale_jitter": False, "aug_config": {}, "run_test": False, "eval_interval": 1,
            "eval_max_dets": protocol["max_detections"], "early_stopping": False, "checkpoint_interval": 1,
            "tensorboard": False, "wandb": False, "mlflow": False, "progress_bar": None,
            "save_dataset_grids": False, "num_sanity_val_steps": 0, "class_names": config["classes"],
            "accelerator": "gpu", "devices": 1, "sync_bn": False,
            "resume": config.get("resume", {}).get("checkpoint", {}).get("path")}


class Backend:
    def train(self, dataset, config, output, observer):
        if version("rfdetr") != "1.11.0":
            raise RuntimeError("Este backend requer RF-DETR 1.11.0")
        import torch
        from pytorch_lightning.callbacks import Callback
        from rfdetr.config import RFDETRNanoConfig, TrainConfig
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
        from rfdetr.training.callbacks import BestModelCallback
        output.mkdir(parents=True, exist_ok=False)
        manifest, records = prepared_dataset(dataset, config["dataset"].get("source_roots"))
        data = export_dataset(records, manifest["sources"], dataset.classes, output / "export", "roboflow_coco")
        train_config = TrainConfig(**options(config, data, output))
        model_config = RFDETRNanoConfig(pretrain_weights=config["model"]["weights"]["path"],
            resolution=config["protocol"]["resolution"], num_classes=len(dataset.classes),
            device=f"cuda:{config['runtime']['device']}", num_select=config["protocol"]["max_detections"],
            model_name="RFDETRNano", compile=False)
        atomic_json(output / "effective_args.json", {"model": model_config.model_dump(mode="json"),
                                                    "training": train_config.model_dump(mode="json")})
        steps = OptimizerSteps()

        class CommonSelection(Callback):
            def on_train_start(self, trainer, module):
                for optimizer in trainer.optimizers:
                    steps.attach(optimizer)
                    observer.bind_optimizer(optimizer, len(trainer.train_dataloader))
                if hasattr(self, "saved_rng"):
                    restore_rng(self.saved_rng)

            def on_save_checkpoint(self, trainer, module, checkpoint):
                checkpoint["ntv_rng"] = rng_state()

            def on_load_checkpoint(self, trainer, module, checkpoint):
                if "ntv_rng" not in checkpoint or not checkpoint.get("optimizer_states"):
                    raise ValueError("Retomada requer checkpoint Lightning completo do núcleo")
                self.saved_rng = checkpoint["ntv_rng"]

            def on_train_epoch_end(self, trainer, module):
                state = (BestModelCallback._get_ema_model_state_dict(trainer, module)
                         if config["protocol"]["weight_policy"] == "ema"
                         else BestModelCallback._get_live_model_state_dict(module))
                candidate = output / "candidate.pth"
                payload = {"model": state, "args": train_config.model_dump(mode="json"),
                           "model_config": BestModelCallback._serialize_model_config(module, state),
                           "model_name": "RFDETRNano", "ntv_classes": config["classes"], "epoch": trainer.current_epoch}
                torch.save(payload, candidate)
                resume = output / f"resume-{trainer.current_epoch:04d}.ckpt"
                trainer.save_checkpoint(resume, weights_only=False)
                if observer.epoch(trainer.current_epoch, candidate, resume):
                    trainer.should_stop = True

        class DataModule(RFDETRDataModule):
            def train_dataloader(self):
                from rfdetr.training.module_data import _worker_init_fn
                # The native tiny-dataset sampler draws with replacement to force
                # five batches. Our protocol visits each training sample once.
                return torch.utils.data.DataLoader(self._require_dataset(self._dataset_train, "fit"),
                    batch_size=self._resolve_batch_size(), shuffle=True, drop_last=False,
                    collate_fn=self._collate_fn_train, num_workers=self._num_workers,
                    pin_memory=self._pin_memory, persistent_workers=self._persistent_workers,
                    prefetch_factor=self._prefetch_factor, worker_init_fn=_worker_init_fn)

        module = RFDETRModelModule(model_config, train_config)
        datamodule = DataModule(model_config, train_config)
        trainer = build_trainer(train_config, model_config, accelerator="gpu", devices=f"{config['runtime']['device']},",
                                deterministic="warn", enable_model_summary=False)
        configure_precision()
        if config["training"]["amp"]:
            trainer.precision_plugin.scaler = torch.amp.GradScaler("cuda", init_scale=config["training"].get("amp_initial_scale", 256))
        # Native metrics remain diagnostic. The shared selector owns the best artifact;
        # removing native promotion also prevents its on_fit_end from replacing weights.
        trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, BestModelCallback)]
        if config['training'].get('control'):
            from pytorch_lightning.callbacks import ModelCheckpoint
            trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)]
        trainer.callbacks.append(CommonSelection())
        trainer.fit(module, datamodule, ckpt_path=train_config.resume, weights_only=False)
        return TrainingResult({"resume": observer.resume}, {},
                              {"optimizer_steps": steps.count, "weight_policy": config["protocol"]["weight_policy"],
                               "effective_batch": train_config.batch_size * train_config.grad_accum_steps,
                               "run_test": False, "package_version": "1.11.0"})

    def load_predictor(self, checkpoint, config):
        if version("rfdetr") != "1.11.0":
            raise RuntimeError("Este backend requer RF-DETR 1.11.0")
        import torch
        from rfdetr import RFDETR
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("ntv_classes") != config["classes"] or state.get("model_name") != "RFDETRNano":
            raise ValueError("Checkpoint RF-DETR incompatível com classes ou variante")
        del state
        model = RFDETR.from_checkpoint(str(checkpoint), trust_checkpoint=True,
            device=f"cuda:{config['runtime']['device']}", resolution=config["protocol"]["resolution"])
        def foreground(module, args, kwargs):
            return (foreground_outputs(args[0], config['classes']), *args[1:]), kwargs
        model.model.postprocess.register_forward_pre_hook(foreground, with_kwargs=True)
        timer = ForwardTimer(model.model.model)
        protocol = config["protocol"]
        def predict(image):
            result = model.predict(image, threshold=protocol["confidence"])
            return detections(result.xyxy.tolist(), result.confidence.tolist(), result.class_id.tolist(),
                config["classes"], image.size, protocol["confidence"], protocol["max_detections"], clip=True), timer.seconds
        return predict

    def predict(self, checkpoint, images, config):
        return predict_samples(self.load_predictor(checked_artifact(checkpoint), config), list(images), config["sources"])[0]
