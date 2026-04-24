from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


FAILURE_CLASSES = {
    "invalid_job_json",
    "unknown_template",
    "missing_template_file",
    "invalid_variable",
    "setup_definition_failure",
    "hfss_start_failure",
    "geometry_failure",
    "mesh_failure",
    "solve_timeout",
    "export_failure",
    "metric_extraction_failure",
    "unexpected_exception",
}


@dataclass
class WorkerError(Exception):
    failure_class: str
    message: str

    def __post_init__(self) -> None:
        if self.failure_class not in FAILURE_CLASSES:
            self.failure_class = "unexpected_exception"

    def __str__(self) -> str:
        return self.message


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_posix_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
