from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.io_utils import read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete local result JSON/artifact files for test jobs while keeping experiment jobs."
    )
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=REPO_ROOT / "queue",
        help="Queue root directory containing done/failed/results.",
    )
    parser.add_argument(
        "--job-prefix",
        default="test_",
        help="Only jobs with job_id starting with this prefix are deleted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be deleted without deleting them.",
    )
    return parser.parse_args()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _collect_result_files(queue_root: Path) -> list[Path]:
    result_files: list[Path] = []
    for state in ("done", "failed"):
        state_dir = queue_root / state
        if state_dir.exists():
            result_files.extend(state_dir.glob("*.result.json"))
    return sorted(result_files)


def _collect_artifacts_from_result(queue_root: Path, result_path: Path) -> list[Path]:
    try:
        payload = read_json(result_path)
    except Exception:
        return []
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return []

    collected: list[Path] = []
    for key in ("touchstone", "csv", "log"):
        rel = artifacts.get(key)
        if not isinstance(rel, str) or not rel.strip():
            continue
        artifact_path = queue_root / rel
        if _is_within(artifact_path, queue_root):
            collected.append(artifact_path)
        else:
            print(f"skip_outside_queue_root={artifact_path}")
    return collected


def _job_id_for_result(result_path: Path) -> str:
    try:
        payload = read_json(result_path)
        value = payload.get("job_id")
        if isinstance(value, str) and value:
            return value
    except Exception:
        pass
    return result_path.name.replace(".result.json", "")


def _delete_file(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        print(f"would_delete={path}")
        return True
    path.unlink()
    print(f"deleted={path}")
    return True


def cleanup_results(queue_root: Path, job_prefix: str, dry_run: bool) -> int:
    if not queue_root.exists():
        print(f"queue_root_missing={queue_root}")
        return 0

    matched_results = 0
    deleted_files = 0
    seen: set[Path] = set()

    for result_file in _collect_result_files(queue_root):
        job_id = _job_id_for_result(result_file)
        if not job_id.startswith(job_prefix):
            continue
        matched_results += 1
        targets = [result_file] + _collect_artifacts_from_result(queue_root, result_file)
        for target in targets:
            if target in seen:
                continue
            seen.add(target)
            if _delete_file(target, dry_run):
                deleted_files += 1

    mode = "dry_run" if dry_run else "apply"
    print(f"mode={mode}")
    print(f"matched_results={matched_results}")
    print(f"deleted_files={deleted_files}")
    return 0


def main() -> int:
    args = parse_args()
    return cleanup_results(
        queue_root=args.queue_root,
        job_prefix=args.job_prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
