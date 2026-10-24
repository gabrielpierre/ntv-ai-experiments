import contextlib
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from ntv_ai.storage import atomic_json, digest, file_hash


@contextlib.contextmanager
def gpu_slot(device, lease_fd=None):
    import fcntl
    import tempfile
    # A worker owns this lock, even if its launcher dies. Different run directories
    # still share the physical GPU and must not overlap their measurements.
    path = Path(tempfile.gettempdir()) / f"ntv-ai-{os.getuid()}-gpu-{device}.lock"
    if lease_fd is not None:
        # The supervisor's open-file description remains locked in each child,
        # including after SSH loss or supervisor death. Never unlock it here.
        expected, actual = path.stat(), os.fstat(lease_fd)
        if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
            raise RuntimeError("Invalid inherited GPU lease")
        fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lease_fd
        return
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield lock.fileno()


def offline_environment(output):
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      YOLO_CONFIG_DIR=str(output / "ultralytics_settings"))


def configure_precision():
    import torch
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_implementation(config):
    if importlib.metadata.version("torch") != "2.10.0+cu128" or importlib.metadata.version("torchvision") != "0.25.0+cu128":
        raise RuntimeError("Ambiente requer torch 2.10.0+cu128 e torchvision 0.25.0+cu128")
    package = config["model"].get("package")
    if package:
        dist = importlib.metadata.distribution(package["name"])
        if dist.version != package["version"]:
            raise RuntimeError("Versão instalada diverge da implementação fixada")


@contextlib.contextmanager
def runtime(config, output):
    offline_environment(output)
    import numpy as np
    import torch
    device = config["runtime"]["device"]
    seed = config["training"]["seed"]
    if config["runtime"].get("cpu_threads"):
        torch.set_num_threads(config["runtime"]["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("Esta receita requer CUDA")
    torch.cuda.set_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    configure_precision()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    code_root = Path(__file__).resolve().parents[1]
    code_digest = digest({str(p.relative_to(code_root)): file_hash(p) for p in sorted(code_root.rglob("*.py"))})
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True)
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "code_digest": code_digest, "nvidia_smi": gpu.stdout.strip(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "seed": seed, "batch": config["training"]["batch"],
            "cpu_threads": torch.get_num_threads(),
            "effective_batch": config["training"]["batch"], "amp": config["training"]["amp"],
            "amp_initial_scale": config["training"].get("amp_initial_scale", 256),
            "inference_precision": "fp32", "inference_batch": 1,
            "tf32": False,
            "box_boundary_policy": "clip to sample bounds; discard degenerate boxes; reject nonfinite outputs",
            "determinism": "seeded; deterministic algorithms warn on unsupported operators",
            "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                        for d in importlib.metadata.distributions() if d.metadata["Name"]))}
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
    (output / "pip-freeze.txt").write_text(freeze.stdout)
    def blocked(*args, **kwargs):
        raise RuntimeError("Aquisição implícita bloqueada; use a operação acquire antes de treinar")
    try:
        with patch("urllib.request.urlopen", blocked), patch("requests.sessions.Session.request", blocked), \
                patch("torch.hub.download_url_to_file", blocked):
            yield info
    finally:
        info.update(duration_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        atomic_json(output / "runtime.json", info, replace=True)


class OptimizerSteps:
    def __init__(self):
        self.count = 0

    def attach(self, optimizer):
        optimizer.register_step_post_hook(self._step)

    def _step(self, optimizer, args, kwargs):
        if float(getattr(optimizer, 'found_inf', 0)) == 0:
            self.count += 1


def rng_state():
    import numpy as np
    import torch
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


class ForwardTimer:
    def __init__(self, model):
        self.seconds = 0.0
        self.handles = [model.register_forward_pre_hook(self.before), model.register_forward_hook(self.after)]

    def before(self, *args):
        import torch
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def after(self, *args):
        import torch
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.started

    def close(self):
        for handle in self.handles:
            handle.remove()
