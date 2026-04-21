from __future__ import annotations

from pathlib import Path

from common.io_utils import read_yaml
from worker.schemas import TemplatesConfig, WorkerConfig


def load_worker_config(path: Path) -> WorkerConfig:
    return WorkerConfig.model_validate(read_yaml(path))


def load_templates_config(path: Path) -> TemplatesConfig:
    return TemplatesConfig.model_validate(read_yaml(path))
