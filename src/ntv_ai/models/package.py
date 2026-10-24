"""Read and restore checkpoint packages without importing training frameworks."""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import shutil

from ntv_ai.experiments.recipes import LAYERS, compatibility, validate_recipe
from ntv_ai.storage import atomic_json, load_json, safe_output


PROTOCOLS = {"full640": {"resolution": 640, "tiling": False},
             "tiles1280": {"resolution": 1280, "tiling": True, "size": 1280, "overlap": .25}}
METADATA = {"config.json", "status.json", "result.json", "environment.json", "selection/selection.json"}


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Caminho relativo inválido no pacote")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError(f"Caminho relativo inválido no pacote: {value}")
    return path


def relocate(value, mappings):
    """Only relocate absolute path components, never substrings or class names."""
    if isinstance(value, dict):
        return {k: relocate(v, mappings) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, mappings) for v in value]
    if isinstance(value, str) and Path(value).is_absolute():
        for old, new in mappings:
            if Path(value).is_relative_to(old):
                return str(new / Path(value).relative_to(old))
    return value


class ModelPackage:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.index = self.read("retomada/indice.json")
        if self.index.get("schema_version") != 1 or not self.index.get("runs"):
            raise ValueError("Índice de modelos não suportado; esperado retomada/indice.json v1")
        ids = set()
        for row in self.index["runs"]:
            if row["id"] in ids:
                raise ValueError("Execução duplicada no pacote")
            ids.add(row["id"])
            destinations, roles, metadata = set(), [], set()
            for ref in row["files"]:
                name = ref["package_path"]
                relative_path(name)
                target = ref["run_relative_path"]
                relative_path(target)
                if target in destinations:
                    raise ValueError("Destino duplicado na restauração")
                destinations.add(target)
                if ref["role"] == "metadata":
                    metadata.add(target)
                else:
                    roles.append(ref["role"])
            if metadata != METADATA or sorted(roles) != ["best_inference", "control", "resume"]:
                raise ValueError("Arquivos obrigatórios de retomada ausentes ou inválidos")

    def path(self, name):
        relative = relative_path(name)
        path = self.root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"Link simbólico não permitido no pacote: {name}")
        return path

    def file(self, name):
        path = self.path(name)
        if not path.is_file():
            raise ValueError(f"Arquivo ausente no pacote: {name}")
        return path

    def read(self, name):
        return load_json(self.file(name))

    @staticmethod
    def reference(row, role):
        return next(ref for ref in row["files"] if ref["role"] == role)

    def metadata(self, row, name):
        ref = next(ref for ref in row["files"] if ref["run_relative_path"] == name)
        return self.read(ref["package_path"])

    def config(self, row):
        config = self.metadata(row, "config.json")
        validate_recipe(config)
        protocol = PROTOCOLS.get(row["protocol"])
        classes = config["dataset"].get("classes", [])
        if (not protocol or config["protocol"]["resolution"] != protocol["resolution"]
                or config["model"]["backend"] != row["backend"] or config["training"]["seed"] != row["seed"]
                or not classes or len(classes) != len(set(classes))):
            raise ValueError("Classes, modelo, seed ou protocolo divergem do índice")
        return config

    def list_models(self):
        rows, keys = [], set()
        for row in self.index["runs"]:
            config = self.config(row)
            model = f"{row['backend']}-{config['model']['variant']}"
            key = (model, row["protocol"], row["seed"])
            if key in keys:
                raise ValueError("Seleção ambígua de modelo/protocolo/seed")
            keys.add(key)
            rows.append({"id": row["id"], "model": model, "protocol": row["protocol"], "seed": row["seed"],
                         "classes": config["dataset"]["classes"],
                         "checkpoint": self.reference(row, "best_inference")["package_path"],
                         "training_stopped": row.get("control_stopped")})
        return {"package": str(self.root), "models": rows}

    def select(self, model, protocol, seed):
        rows = [r for r in self.list_models()["models"]
                if (r["model"], r["protocol"], r["seed"]) == (model, protocol, seed)]
        if len(rows) != 1:
            raise ValueError("Modelo/protocolo/seed não encontrado; consulte models list")
        return next(r for r in self.index["runs"] if r["id"] == rows[0]["id"])

    def validate_run(self, row):
        config = self.config(row)
        result = self.metadata(row, "result.json")
        status = self.metadata(row, "status.json")
        selection = self.metadata(row, "selection/selection.json")
        control_ref = self.reference(row, "control")
        control = self.read(control_ref["package_path"])
        for role, saved in (("resume", result["resume"]), ("control", result["control"]),
                            ("best_inference", result["best"]["checkpoint"])):
            ref = self.reference(row, role)
            if Path(saved["path"]) != Path(row["original_run_path"]) / ref["run_relative_path"]:
                raise ValueError("Referências de checkpoints inconsistentes")
        same_selection = all(selection["best"][key] == result["best"][key] for key in ("epoch", "mAP50_95"))
        same_checkpoint = selection["best"]["checkpoint"]["path"] == result["best"]["checkpoint"]["path"]
        if (status["run_id"] != row["run_id"] or not same_selection or not same_checkpoint
                or control["config"] != config["training"].get("control")
                or control["state"]["epoch"] != result["last_epoch"]
                or control["state"]["stopped"] != row.get("control_stopped")):
            raise ValueError("Estado de controle ou seleção incompatível com a execução")
        return config, control

    def restore(self, row, output, storage_root, source_roots=None):
        config, control = self.validate_run(row)
        sources = [(ref, self.file(ref["package_path"])) for ref in row["files"]]
        output = safe_output(Path(output), [self.root])
        old_root, old_run = Path(self.index["source_root"]), Path(row["original_run_path"])
        if (not old_root.is_absolute() or not old_run.is_relative_to(old_root)
                or old_root == Path("/") or old_run == old_root):
            raise ValueError("Raiz original inválida no índice")
        # Never overwrite an existing run or an original artifact, including an empty directory.
        output.mkdir(parents=True, exist_ok=False)
        try:
            for ref, source in sources:
                target = output / ref["run_relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            mappings = [(old_run, output), (old_root, Path(storage_root).resolve())]
            restored = relocate(config, mappings)
            restored["runtime"]["storage_root"] = str(Path(storage_root).resolve())
            manifest_path = Path(restored["dataset"]["manifest"]["path"])
            if manifest_path.is_file():
                manifest = load_json(manifest_path)
                roots = {k: relocate(v["root"], mappings) for k, v in manifest["sources"].items()}
                restored["dataset"]["source_roots"] = {**roots, **restored["dataset"].get("source_roots", {})}
            if source_roots:
                restored["dataset"].setdefault("source_roots", {}).update(source_roots)
            if compatibility(config) != compatibility(restored):
                raise ValueError("A restauração alterou a identidade do experimento")
            atomic_json(output / "config.json", restored, replace=True)
            for name in ("result.json", "status.json", "selection/selection.json"):
                atomic_json(output / name, relocate(load_json(output / name), mappings), replace=True)
            recipe = {key: deepcopy(restored[key]) for key in ("schema_version", "purpose", *LAYERS)}
            atomic_json(output / "recipe.json", recipe)
            required = [restored["dataset"]["manifest"]["path"], restored["model"]["weights"]["path"]]
            if restored["model"].get("implementation"):
                required.append(restored["model"]["implementation"]["path"])
            report = {"source_package": str(self.root), "source_run": row["run_id"],
                      "files": row["files"],
                      "restored_run": str(output), "recipe": str(output / "recipe.json"),
                      "training_stopped": control["state"]["stopped"],
                      "missing_artifacts": [p for p in required if not Path(p).exists()],
                      "resume_policy": "early stopping preserved; completed training cannot reset patience"}
            atomic_json(output / "restoration.json", report)
            return report
        except BaseException:
            shutil.rmtree(output)
            raise
