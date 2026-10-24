import copy
import re
import math
from pathlib import Path

from ntv_ai.experiments.config import read_config
from ntv_ai.models.registry import specification
from ntv_ai.storage import digest, file_hash

LAYERS = ("dataset", "protocol", "model", "training", "runtime")


def resolve_recipe(path: Path, storage_root=None):
    value = read_config(path)
    unknown = value.keys() - {*LAYERS, "schema_version", "purpose"}
    if unknown or value.get("schema_version") != 1:
        raise ValueError(f"Receita v1 inválida: {sorted(unknown)}")
    result = copy.deepcopy(value)
    evidence = {}
    for key in LAYERS:
        layer = value[key]
        if isinstance(layer, str):
            source = (path.parent / layer).resolve()
            result[key] = read_config(source)
            evidence[key] = file_hash(source)
        elif not isinstance(layer, dict):
            raise ValueError(f"Camada inválida: {key}")
    root = Path(storage_root or result["runtime"].get("storage_root", "."))
    if not root.is_absolute():
        root = path.parent / root
    result["runtime"]["storage_root"] = str(root.resolve())
    validate_recipe(result)
    result["provenance"] = {"recipe_sha256": file_hash(path), "layers": evidence}
    return result


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} deve ser inteiro >= {minimum}")


def validate_recipe(config):
    model, protocol, training, runtime = (config[k] for k in ("model", "protocol", "training", "runtime"))
    spec = specification(model["backend"])
    native = training.get("native", {})
    if not isinstance(native, dict) or native.keys() - set(spec.native_options):
        raise ValueError("Opções nativas desconhecidas ou controladas pelo protocolo")
    for key, value in native.items():
        if key == "optimizer":
            if value not in ("SGD", "Adam", "AdamW", "RMSProp"):
                raise ValueError("Otimizador deve ser explícito; auto não é permitido")
        elif key == "augment":
            if type(value) is not bool:
                raise ValueError("augment deve ser booleano")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Valor nativo inválido: {key}")
        elif key in ("mosaic", "mixup", "cutmix", "copy_paste", "flipud", "fliplr", "momentum", "hsv_h", "hsv_s", "hsv_v") and value > 1:
            raise ValueError(f"{key} deve estar entre 0 e 1")
        elif key in ("warmup_steps", "close_mosaic") and type(value) is not int:
            raise ValueError(f"{key} deve ser inteiro")
    if runtime.keys() - {"environment", "device", "storage_root", "runs_root", "python", "cpu_threads"}:
        raise ValueError("Opção desconhecida no ambiente de execução")
    if model["variant"] not in spec.variants:
        raise ValueError("Variante não suportada")
    if config["purpose"] not in ("software_smoke_only", "scientific", "exploratory"):
        raise ValueError("Finalidade inválida")
    if config["purpose"] == "scientific" and config["dataset"].get("fixture"):
        raise ValueError("Fixture de software não pode ser usado como benchmark científico")
    if model.keys() - {"backend", "variant", "weights", "implementation", "package", "pretraining"}:
        raise ValueError("Opção desconhecida no modelo")
    if not model.get("weights") or not model.get("pretraining", "unspecified"):
        raise ValueError("Referência de pesos obrigatória")
    for key in ("resolution", "max_detections"):
        positive_int(protocol[key], key, 10 if key == "max_detections" else 1)
    if protocol["resolution"] % spec.resolution_multiple:
        raise ValueError(f"Resolução deve ser múltipla de {spec.resolution_multiple}")
    if protocol.get("checkpoint_metric") != "mAP50_95" or protocol.get("selection_split") != "val":
        raise ValueError("Seleção requer mAP50_95 comum no split val")
    if protocol.get("weight_policy") not in spec.weight_policies:
        raise ValueError("Política de pesos incompatível com o backend")
    if not 0 <= protocol["confidence"] <= 1 or not 0 < protocol["tile_iou"] <= 1:
        raise ValueError("Limiares inválidos")
    allowed_protocol = {"id", "resolution", "max_detections", "checkpoint_metric", "selection_split",
                        "weight_policy", "confidence", "tile_iou", "warmup", "repeats"}
    if protocol.keys() - allowed_protocol:
        raise ValueError("Opção desconhecida no protocolo")
    for key in ("warmup", "repeats"):
        if key in protocol:
            positive_int(protocol[key], key)
    for key in ("epochs", "batch", "workers", "seed"):
        positive_int(training[key], key, 0 if key in ("workers", "seed") else 1)
    if type(training["amp"]) is not bool or training.keys() - {"epochs", "batch", "workers", "seed", "amp", "amp_initial_scale", "native", "control"}:
        raise ValueError("Receita de treinamento inválida")
    if "control" in training:
        from ntv_ai.experiments.control import validate_control
        validate_control(training["control"])
        if model['backend'] == 'yolo26':
            positive_int(native.get('loss_horizon'), 'loss_horizon')
    positive_int(training.get("amp_initial_scale", 256), "amp_initial_scale")
    if type(runtime["device"]) is not int or runtime["device"] < 0:
        raise ValueError("Informe um único índice de GPU")
    if not runtime.get("environment"):
        raise ValueError("Ambiente de execução obrigatório")
    if "cpu_threads" in runtime:
        positive_int(runtime["cpu_threads"], "cpu_threads")


def local_reference(ref, root):
    path = (Path(root) / ref["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"Arquivo ausente: {path}")
    return str(path)


def checked_reference(ref, root):
    from ntv_ai.models.common import checked_artifact
    from ntv_ai.domain.records import ArtifactReference
    if not re.fullmatch(r"[a-f0-9]{64}", ref.get("sha256", "")):
        raise ValueError("Artefato requer SHA256 fixado")
    path = Path(root) / ref["path"]
    checked_artifact(ArtifactReference(str(path), ref["sha256"]))
    return str(path.resolve())


def compatibility(config):
    """Paths are locators; all scientific inputs remain in the identity, except seed."""
    value = copy.deepcopy(config)
    value.pop("provenance", None)
    value.pop("resume", None)
    # A learned checkpoint is an outcome of the seed, not an initial condition.
    # Its exact hash remains in the run config, outside the aggregation identity.
    value.pop('inference_checkpoint', None)
    value["training"].pop("seed", None)
    runtime = value.pop("runtime")
    value["execution"] = {k: v for k, v in runtime.items() if k not in ("storage_root", "runs_root", "python", "cpu_threads")}
    dataset = value["dataset"]
    dataset.pop("source_roots", None)
    if "manifest" in dataset:
        dataset["manifest"].pop("path", None)
    for ref in (value["model"].get("weights"), value["model"].get("implementation")):
        if isinstance(ref, dict):
            ref.pop("path", None)
    wheel = value["model"].get("package", {}).get("wheel")
    if wheel:
        wheel.pop("path", None)
    return digest(value)
