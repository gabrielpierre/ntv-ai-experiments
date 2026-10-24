from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class BackendSpec:
    module: str
    variants: tuple[str, ...]
    formats: tuple[str, ...]
    resolution_multiple: int = 32
    training: bool = True
    inference: bool = True
    resume: bool = True
    weight_policies: tuple[str, ...] = ("regular", "ema")
    native_options: tuple[str, ...] = ()


BACKENDS = {
    "yolo26": BackendSpec("ntv_ai.models.backends.yolo26", ("n", "s", "m", "l", "x"), ("yolo",),
                          weight_policies=("ema",), native_options=("optimizer", "lr0", "lrf", "warmup_epochs",
                          "mosaic", "close_mosaic", "momentum", "weight_decay", "hsv_h", "hsv_s", "hsv_v", "degrees",
                          "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mixup", "cutmix", "copy_paste", "loss_horizon")),
    "dfine": BackendSpec("ntv_ai.models.backends.dfine", ("s",), ("coco_zero_based",),
                         native_options=("lr", "lr_backbone", "weight_decay", "warmup_steps", "augment")),
    "rfdetr": BackendSpec("ntv_ai.models.backends.rfdetr", ("nano",), ("roboflow_coco",),
                          native_options=("lr", "lr_encoder", "weight_decay", "warmup_epochs", "clip_max_norm")),
}


def specification(name):
    if name not in BACKENDS:
        raise ValueError(f"Backend desconhecido: {name}; disponíveis: {', '.join(BACKENDS)}")
    return BACKENDS[name]


def backend(name):
    return import_module(specification(name).module).Backend()
