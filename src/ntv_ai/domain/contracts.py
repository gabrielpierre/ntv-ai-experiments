from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .records import ArtifactReference, ImageRecord, PredictionRecord, PreparedDataset, TrainingResult


class DatasetReader(Protocol):
    def read(self) -> list[ImageRecord]: ...


class TrainingBackend(Protocol):
    def train(self, dataset: PreparedDataset, config: dict, output: Path) -> TrainingResult: ...


class InferenceBackend(Protocol):
    def predict(
        self, checkpoint: ArtifactReference, images: Iterable[ImageRecord], config: dict
    ) -> Iterable[PredictionRecord]: ...


class EpochObserver(Protocol):
    def epoch(self, epoch: int, checkpoint: Path, resume: Path) -> None: ...


class ExperimentTrainer(Protocol):
    def train(self, dataset: PreparedDataset, config: dict, output: Path,
              observer: EpochObserver) -> TrainingResult: ...


class SamplePredictor(Protocol):
    def __call__(self, image) -> tuple[list, float]: ...


class ReloadableInferenceBackend(InferenceBackend, Protocol):
    def load_predictor(self, checkpoint: Path, config: dict) -> SamplePredictor: ...
