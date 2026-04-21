from __future__ import annotations

import json
from pathlib import Path


def _looks_complete_json(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def try_claim_one_job(queue_root: Path) -> Path | None:
    pending_dir = queue_root / "pending"
    running_dir = queue_root / "running"
    running_dir.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)

    for candidate in sorted(pending_dir.glob("*.json")):
        if not _looks_complete_json(candidate):
            continue
        destination = running_dir / candidate.name
        try:
            candidate.rename(destination)
            return destination
        except OSError:
            continue
    return None
