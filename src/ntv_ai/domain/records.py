from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class Geometry:
    type: str
    coordinates: list
    convention: str = "pixel_edges_xyxy"


@dataclass
class Annotation:
    annotation_id: str
    original_label: str
    geometry: Geometry
    source: dict[str, Any]
    label: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    original: dict[str, Any] = field(default_factory=dict)
    mapping_status: str = "unmapped"
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageRecord:
    image_id: str
    dataset_id: str
    path: str
    width: int = 0
    height: int = 0
    sha256: str = ""
    pixel_sha256: str = ""
    annotations: list[Annotation] = field(default_factory=list)
    annotation_scope: dict[str, str] = field(default_factory=dict)
    group_id: str | None = None
    split: str | None = None
    original_splits: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ImageRecord":
        data = dict(value)
        data["annotations"] = [
            Annotation(**{**a, "geometry": Geometry(**a["geometry"])})
            for a in value.get("annotations", [])
        ]
        return cls(**data)


@dataclass
class Issue:
    code: str
    source: str
    message: str
    image_id: str | None = None
    annotation_id: str | None = None


@dataclass
class Detection:
    class_id: str
    bbox_xyxy: list[float]
    score: float


@dataclass
class PredictionRecord:
    image_id: str
    detections: list[Detection]
    status: str = "processed"


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str


@dataclass(frozen=True)
class PreparedDataset:
    manifest: ArtifactReference
    view_id: str
    classes: tuple[str, ...]


@dataclass
class TrainingResult:
    checkpoints: dict[str, ArtifactReference]
    native_metrics: dict[str, Any]
    metadata: dict[str, Any]
