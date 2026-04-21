from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML: {path_obj}")
    return data


def read_json(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")
        f.flush()
    tmp_path.replace(path_obj)
