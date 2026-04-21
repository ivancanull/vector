from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


QUEUE_SUBDIRS = ("pending", "running", "done", "failed", "results")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    queue: Path
    pending: Path
    running: Path
    done: Path
    failed: Path
    results: Path
    database: Path
    experiments_db: Path


def ensure_runtime_dirs(queue_root: Path, logs_dir: Path) -> None:
    for subdir in QUEUE_SUBDIRS:
        (queue_root / subdir).mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "hfss_runs").mkdir(parents=True, exist_ok=True)


def default_paths(root: str | Path = ".") -> ProjectPaths:
    root_path = Path(root).resolve()
    queue = root_path / "queue"
    database = root_path / "database"
    return ProjectPaths(
        root=root_path,
        queue=queue,
        pending=queue / "pending",
        running=queue / "running",
        done=queue / "done",
        failed=queue / "failed",
        results=queue / "results",
        database=database,
        experiments_db=database / "experiments.db",
    )
