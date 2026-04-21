from __future__ import annotations

from pathlib import Path


QUEUE_SUBDIRS = ("pending", "running", "done", "failed", "results")


def ensure_runtime_dirs(queue_root: Path, logs_dir: Path) -> None:
    for subdir in QUEUE_SUBDIRS:
        (queue_root / subdir).mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "hfss_runs").mkdir(parents=True, exist_ok=True)
