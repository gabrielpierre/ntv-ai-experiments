from pathlib import Path

import yaml


def read_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Configuração precisa ser um objeto: {path}")
    return value


def reference(config_path: Path, value: str) -> Path:
    return (config_path.resolve().parent / value).resolve()
