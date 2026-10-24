import importlib.metadata
import logging
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ntv_ai.storage import atomic_json, digest, file_hash


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Run:
    def __init__(self, root: Path, operation: str, config: dict):
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        self.started = time.monotonic()
        self.path = root / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        code_root = Path(__file__).resolve().parents[1]
        hashes = {str(p.relative_to(code_root)): file_hash(p) for p in sorted(code_root.rglob("*.py"))}
        result = subprocess.run(["git", "-C", str(code_root), "rev-parse", "HEAD"], capture_output=True, text=True) if self._has_git() else None
        environment = {
            "python": sys.version, "platform": platform.platform(),
            "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
            "code_sha": result.stdout.strip() if result and result.returncode == 0 else None,
            "code_digest": digest(hashes), "code_files": hashes,
            "device": "cpu", "gpu": None, "vram": None,
        }
        atomic_json(self.path / "config.json", config)
        atomic_json(self.path / "environment.json", environment)
        self.status = {"run_id": self.id, "operation": operation, "state": "running", "started_at": now(), "artifacts": {}}
        atomic_json(self.path / "status.json", self.status)
        self.logger = logging.getLogger(f"ntv_ai.run.{self.id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = logging.FileHandler(self.path / "run.log", encoding="utf-8")
        self.logger.addHandler(self.handler)

    @staticmethod
    def _has_git() -> bool:
        import shutil
        return shutil.which("git") is not None

    def __enter__(self) -> "Run":
        self.logger.info("Execução iniciada: %s", self.status["operation"])
        return self

    def __exit__(self, kind, value, traceback) -> bool:
        state = "completed" if kind is None else ("interrupted" if issubclass(kind, KeyboardInterrupt) else "failed")
        self.status.update(state=state, finished_at=now(), duration_seconds=time.monotonic()-self.started)
        if value is not None:
            self.status["error"] = str(value)
            self.logger.error("Falha", exc_info=(kind, value, traceback))
        self.logger.info("Estado final: %s", state)
        atomic_json(self.path / "status.json", self.status, replace=True)
        self.logger.removeHandler(self.handler)
        self.handler.close()
        return False
