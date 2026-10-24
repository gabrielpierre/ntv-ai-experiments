from pathlib import Path

from PIL import Image

from ntv_ai.data.audit import audit_records
from ntv_ai.data.views import eligibility
from ntv_ai.domain.records import ArtifactReference, PreparedDataset
from ntv_ai.storage import file_hash, inside, read_snapshot


def artifact(path: Path) -> ArtifactReference:
    return ArtifactReference(str(path.resolve()), file_hash(path))


def checked_artifact(ref: ArtifactReference) -> Path:
    path = Path(ref.path).resolve()
    if file_hash(path) != ref.sha256:
        raise ValueError(f"Hash divergente: {path}")
    return path


def prepared_dataset(dataset: PreparedDataset, source_roots=None):
    path = checked_artifact(dataset.manifest)
    manifest, records = read_snapshot(path)
    for key, root in (source_roots or {}).items():
        if key not in manifest["sources"]:
            raise ValueError(f"Fonte desconhecida: {key}")
        manifest["sources"][key]["root"] = str(Path(root).resolve())
    view = manifest.get("view") or {}
    if not manifest["training_ready"] or not view.get("eligible"):
        raise ValueError("Manifesto inelegível para treinamento")
    if view.get("id") != dataset.view_id or tuple(view.get("classes", [])) != dataset.classes:
        raise ValueError("Classes ou view incompatíveis com o manifesto")
    index = {r.image_id: r for r in records}
    ids = view["image_ids"]
    if len(ids) != len(set(ids)) or set(ids) - index.keys():
        raise ValueError("Referências de imagens inválidas na view")
    selected = [index[key] for key in ids]
    if any(r.metadata.get("excluded") for r in selected) or eligibility(selected, list(dataset.classes)):
        raise ValueError("A view contém pendências de elegibilidade")
    report = audit_records(manifest, records)
    if any(i["code"].endswith("integrity_mismatch") for i in report["issues"]):
        raise ValueError("Integridade das fontes divergente")
    return manifest, selected


def verify_framework_images(records, sources):
    for record in records:
        path = inside(Path(sources[record.dataset_id]["root"]), record.path)
        with Image.open(path) as image:
            image.verify()
            is_jpeg = image.format == "JPEG"
        if is_jpeg:
            with path.open("rb") as stream:
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    raise ValueError(f"JPEG truncado; reparação automática bloqueada: {record.image_id}")
