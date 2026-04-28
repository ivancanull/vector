from __future__ import annotations

"""Clean queue directories for a fresh server/worker run.

Default mode is dry-run; pass --apply to delete files.
"""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_ROOT = REPO_ROOT / "queue"

TARGET_SUBDIRS = [
    "pending",
    "running",
    "done",
    "failed",
    "results",
    "archive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean queue directories (pending/running/done/failed/results/archive).")
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=DEFAULT_QUEUE_ROOT,
        help="Queue root directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, script runs in dry-run mode.",
    )
    return parser.parse_args()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _iter_targets(queue_root: Path) -> list[Path]:
    targets: list[Path] = []
    for name in TARGET_SUBDIRS:
        subdir = queue_root / name
        if not subdir.exists():
            continue
        for path in subdir.rglob("*"):
            if path.is_file():
                targets.append(path)
    return sorted(targets)


def _delete_empty_dirs(root: Path, dry_run: bool) -> int:
    deleted = 0
    # Deepest-first so nested dirs are removed before parents.
    dirs = sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        if any(d.iterdir()):
            continue
        if dry_run:
            print(f"would_rmdir={d}")
            deleted += 1
            continue
        d.rmdir()
        print(f"rmdir={d}")
        deleted += 1
    return deleted


def cleanup_queue(queue_root: Path, *, apply: bool) -> int:
    dry_run = not apply

    if not queue_root.exists():
        print(f"queue_root_missing={queue_root}")
        return 0

    files = _iter_targets(queue_root)
    removed_files = 0

    for path in files:
        if not _is_within(path, queue_root):
            print(f"skip_outside_queue_root={path}")
            continue

        if dry_run:
            print(f"would_delete={path}")
            removed_files += 1
            continue

        path.unlink()
        print(f"deleted={path}")
        removed_files += 1

    removed_dirs = _delete_empty_dirs(queue_root, dry_run=dry_run)
    mode = "apply" if apply else "dry_run"
    print(f"mode={mode}")
    print(f"queue_root={queue_root}")
    print(f"removed_files={removed_files}")
    print(f"removed_dirs={removed_dirs}")
    return 0


def main() -> int:
    args = parse_args()
    return cleanup_queue(args.queue_root, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
