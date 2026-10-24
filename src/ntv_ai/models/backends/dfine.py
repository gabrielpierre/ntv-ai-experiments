import copy
from contextlib import nullcontext
import sys
from pathlib import Path

from ntv_ai.domain.records import TrainingResult
from ntv_ai.models.common import artifact, checked_artifact, prepared_dataset
from ntv_ai.models.exports import export_dataset
from ntv_ai.models.inference import detections, predict_samples
from ntv_ai.models.runtime import ForwardTimer, OptimizerSteps, restore_rng, rng_state
from ntv_ai.storage import atomic_json

REVISION = "956d1709314c2c6a4df6f34de232054578a7449f"
NATIVE_OPTIONS = {"lr", "lr_backbone", "weight_decay", "warmup_steps", "augment"}


def native_config(config, output, data=None):
    root = Path(config["model"]["implementation"]["path"])
    if config["model"]["implementation"]["revision"] != REVISION:
        raise ValueError("Revisão D-FINE não suportada")
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    from src.core import YAMLConfig
    base = YAMLConfig(str(root / "configs/dfine/dfine_hgnetv2_s_coco.yml"))
    value = copy.deepcopy(base.yaml_cfg)
    protocol, training = config["protocol"], config["training"]
    native = training.get("native", {})
    if native.keys() - NATIVE_OPTIONS:
        raise ValueError(f"Opções D-FINE não suportadas: {sorted(native.keys() - NATIVE_OPTIONS)}")
    size = protocol["resolution"]
    value.update(num_classes=len(config["classes"]), remap_mscoco_category=False,
                 eval_spatial_size=[size, size], device=f"cuda:{config['runtime']['device']}",
                 epochs=training["epochs"], output_dir=str(output / "native"), use_amp=training["amp"],
                 use_ema=protocol["weight_policy"] == "ema", use_wandb=False, sync_bn=False,
                 checkpoint_freq=1, print_freq=1)
    value["HGNetv2"]["pretrained"] = False
    value["scaler"]["init_scale"] = training.get("amp_initial_scale", 256)
    value["DFINEPostProcessor"]["num_top_queries"] = protocol["max_detections"]
    value["optimizer"]["lr"] = native.get("lr", .0002)
    value["optimizer"]["weight_decay"] = native.get("weight_decay", .0001)
    for group in value["optimizer"]["params"]:
        if "lr" in group:
            group["lr"] = native.get("lr_backbone", .0001)
    value["lr_warmup_scheduler"]["warmup_duration"] = native.get("warmup_steps", 0)
    if training.get('control'):
        value['lr_warmup_scheduler']['warmup_duration'] = 0
        value['checkpoint_freq'] = training['epochs'] + 1
    for split in ("train", "val"):
        loader = value[f"{split}_dataloader"]
        loader.update(total_batch_size=training["batch"] if split == "train" else 1,
                      num_workers=training["workers"], drop_last=False)
        transforms = loader["dataset"]["transforms"]
        ops = transforms["ops"]
        if not native.get("augment", False):
            ops = [op for op in ops if op["type"] in ("Resize", "ConvertPILImage", "ConvertBoxes")]
        for op in ops:
            if op["type"] == "Resize":
                op["size"] = [size, size]
        transforms["ops"] = ops
        if "policy" in transforms:
            transforms["policy"]["epoch"] = training["epochs"] + 1
        loader["collate_fn"].update(base_size=size, base_size_repeat=None, stop_epoch=training["epochs"] + 1)
        if data:
            loader["dataset"].update(img_folder="/", ann_file=str(data / f"{split}.json"))
    return YAMLConfig(str(root / "configs/dfine/dfine_hgnetv2_s_coco.yml"), **value)


class Backend:
    def train(self, dataset, config, output, observer):
        import torch
        from unittest.mock import patch
        output.mkdir(parents=True, exist_ok=False)
        manifest, records = prepared_dataset(dataset, config["dataset"].get("source_roots"))
        data = export_dataset(records, manifest["sources"], dataset.classes, output / "export", "coco_zero_based")
        cfg = native_config(config, output, data)
        from src.solver.det_solver import DetSolver
        import src.solver.det_solver as native_solver
        from src.misc import dist_utils
        steps = OptimizerSteps()
        cfg.tuning = config["model"]["weights"]["path"] if not config.get("resume") else None
        cfg.resume = config.get("resume", {}).get("checkpoint", {}).get("path")

        class Solver(DetSolver):
            def train(self):
                super().train()
                steps.attach(self.optimizer)
                if config['training'].get('control'):
                    self.lr_warmup_scheduler = None
                    self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=1)
                observer.bind_optimizer(self.optimizer, len(self.train_dataloader))
                if cfg.resume:
                    state = torch.load(cfg.resume, map_location="cpu", weights_only=False)
                    restore_rng(state["ntv_rng"])

            def load_resume_state(self, path):
                state = torch.load(path, map_location="cpu", weights_only=False)
                if "ntv_rng" not in state:
                    raise ValueError("Retomada requer checkpoint completo do núcleo")
                self.load_state_dict(state)

        solver = Solver(cfg)
        evaluate_native = native_solver.evaluate
        def evaluated(*args, **kwargs):
            result = evaluate_native(*args, **kwargs)
            epoch = args[6] if len(args) > 6 else kwargs["epoch"]
            # Native resume may evaluate its starting checkpoint before training.
            if steps.count:
                state = solver.state_dict()
                state["ntv_rng"] = rng_state()
                resume = output / f"resume-{epoch:04d}.pth"
                torch.save(state, resume)
                module = solver.ema.module if config["protocol"]["weight_policy"] == "ema" else solver.model
                candidate = output / "candidate.pth"
                torch.save({"model": dist_utils.de_parallel(module).state_dict(), "classes": config["classes"],
                            "variant": "s", "revision": REVISION}, candidate)
                if observer.epoch(epoch, candidate, resume):
                    from ntv_ai.experiments.control import CommonEarlyStop
                    raise CommonEarlyStop()
            return result
        effective = {**cfg.yaml_cfg, "tuning": cfg.tuning, "resume": cfg.resume}
        if config['training'].get('control'):
            effective.update(lr_scheduler={'type':'torch.optim.lr_scheduler.ConstantLR',
                                          'factor':1.0, 'total_iters':1},
                             lr_warmup_scheduler=None, common_control=config['training']['control'],
                             native_checkpoint_saving=False)
        atomic_json(output / "effective_args.json", effective)
        # FLOP analysis is unrelated to optimization and allocates extra device memory.
        with patch.object(native_solver, "evaluate", evaluated), patch.object(native_solver, "stats",
                lambda cfg: (sum(p.numel() for p in cfg.model.parameters()), "FLOP analysis disabled")), \
                (patch.object(dist_utils, 'save_on_master', lambda *args, **kwargs: None)
                 if config['training'].get('control') else nullcontext()):
            from ntv_ai.experiments.control import CommonEarlyStop
            try:
                solver.fit()
            except CommonEarlyStop:
                pass
            finally:
                solver.cleanup()
        return TrainingResult({"resume": observer.resume}, {},
                              {"optimizer_steps": steps.count, "weight_policy": config["protocol"]["weight_policy"],
                               "effective_batch": config["training"]["batch"], "external_revision": REVISION})

    def load_predictor(self, checkpoint, config):
        import torch
        import torchvision.transforms as transforms
        cfg = native_config(config, Path(checkpoint).parent)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("classes") != config["classes"] or state.get("revision") != REVISION or state.get("variant") != "s":
            raise ValueError("Checkpoint D-FINE incompatível com classes ou implementação")
        model = cfg.model
        model.load_state_dict(state["model"], strict=True)
        device = f"cuda:{config['runtime']['device']}"
        model.to(device).eval()
        postprocessor = cfg.postprocessor.to(device).eval()
        timer = ForwardTimer(model)
        protocol = config["protocol"]
        transform = transforms.Compose([transforms.Resize((protocol["resolution"], protocol["resolution"])), transforms.ToTensor()])
        def predict(image):
            with torch.inference_mode():
                tensor = transform(image).unsqueeze(0).to(device)
                original_size = torch.tensor([image.size], device=device)
                result = postprocessor(model(tensor), original_size)[0]
                items = detections(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(),
                    result["labels"].cpu().tolist(), config["classes"], image.size,
                    protocol["confidence"], protocol["max_detections"], clip=True)
                return items, timer.seconds
        return predict

    def predict(self, checkpoint, images, config):
        return predict_samples(self.load_predictor(checked_artifact(checkpoint), config), list(images), config["sources"])[0]
