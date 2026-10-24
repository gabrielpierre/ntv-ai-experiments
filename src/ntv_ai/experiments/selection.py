import math
import shutil
from dataclasses import asdict
from pathlib import Path

from ntv_ai.models.common import artifact
from ntv_ai.storage import atomic_json, load_json


class CheckpointSelection:
    def __init__(self, output: Path, parent=None):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)
        self.best = None
        self.history = []
        if parent:
            state = load_json(Path(parent) / "selection.json")
            self.best = state["best"]
            self.history = state["history"]
            src = Path(self.best["checkpoint"]["path"])
            dst = self.output / ("best" + src.suffix)
            shutil.copy2(src, dst)
            self.best = {**self.best, "checkpoint": asdict(artifact(dst))}

    def consider(self, epoch, metric, checkpoint: Path):
        if metric is None or not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError("mAP50_95 comum ausente ou inválido")
        if self.history and epoch <= self.history[-1]["epoch"]:
            raise ValueError("Épocas de seleção devem ser crescentes")
        self.history.append({"epoch": epoch, "mAP50_95": metric})
        if self.best is None or metric > self.best["mAP50_95"]:
            target = self.output / ("best" + checkpoint.suffix)
            shutil.copy2(checkpoint, target)
            self.best = {"epoch": epoch, "mAP50_95": metric, "checkpoint": asdict(artifact(target))}
        atomic_json(self.output / "selection.json", {
            "metric": "mAP50_95", "split": "val", "tie": "first", "best": self.best, "history": self.history,
        }, replace=True)
        return self.best
