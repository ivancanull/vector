from __future__ import annotations

"""Linear local worker for offline filesystem queues."""

import argparse
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.io_utils import read_json  # noqa: E402
from common.paths import ensure_runtime_dirs  # noqa: E402
from worker.config import load_templates_config, load_worker_config  # noqa: E402
from worker.logger import setup_global_logger  # noqa: E402
from worker.template_registry import TemplateRegistry  # noqa: E402
from worker.worker_main import ClaimedJobProcessor  # noqa: E402


DEFAULT_WORKER_CONFIG = REPO_ROOT / "configs" / "worker.yaml"
DEFAULT_QUEUE_ROOT = REPO_ROOT / ".runtime" / "queue"


def _claim_one_pending(queue_root: Path) -> Path | None:
    pending_dir = queue_root / "pending"
    running_dir = queue_root / "running"
    pending_dir.mkdir(parents=True, exist_ok=True)
    running_dir.mkdir(parents=True, exist_ok=True)
    for pending_path in sorted(pending_dir.glob("*.json")):
        claim_lock = pending_path.with_name(f"{pending_path.name}.claim")
        running_path = running_dir / pending_path.name
        try:
            fd = os.open(claim_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except OSError:
            continue
        try:
            if running_path.exists():
                continue
            pending_path.replace(running_path)
            return running_path
        except OSError:
            continue
        finally:
            try:
                claim_lock.unlink()
            except FileNotFoundError:
                pass
    return None


def _load_processor(
    *,
    worker_config_path: Path,
    queue_root: Path,
    logs_dir: Path,
    template_copy_root: Path,
    preserve_local_project_copy: bool,
):
    cfg = load_worker_config(worker_config_path)
    original_queue_root = cfg.queue_root
    original_results_dir = cfg.results_dir
    cfg.queue_root = queue_root
    cfg.results_dir = _resolve_results_dir(
        original_queue_root=original_queue_root,
        original_results_dir=original_results_dir,
        queue_root=queue_root,
    )
    cfg.logs_dir = logs_dir
    cfg.copy_template_per_job = True
    cfg.template_copy_root = template_copy_root
    cfg.preserve_local_project_copy = preserve_local_project_copy
    cfg.close_desktop_after_job = True
    cfg.new_desktop = True
    templates_cfg = load_templates_config(cfg.templates_yaml)
    ensure_runtime_dirs(cfg.queue_root, cfg.logs_dir)
    template_registry = TemplateRegistry(templates_cfg)
    global_logger = setup_global_logger(cfg.logs_dir)
    processor = ClaimedJobProcessor.default(
        cfg=cfg,
        template_registry=template_registry,
        global_logger=global_logger,
    )
    return cfg, processor


def _resolve_results_dir(
    *,
    original_queue_root: Path,
    original_results_dir: Path,
    queue_root: Path,
) -> Path:
    try:
        original_results_dir.resolve().relative_to(original_queue_root.resolve())
    except ValueError:
        return original_results_dir
    return queue_root / "results"


def _job_id_from_path(path: Path) -> str:
    try:
        payload = read_json(path)
        job_id = payload.get("job_id")
        if isinstance(job_id, str) and job_id:
            return job_id
    except Exception:
        pass
    return path.stem


def run(args: argparse.Namespace) -> int:
    queue_root = args.queue_root
    slot_root = args.slot_root or (queue_root.parent / "worker_linear")
    logs_dir = args.logs_dir or (slot_root / "logs")
    template_copy_root = args.template_copy_root or (slot_root / "aedt_copies")
    cfg, processor = _load_processor(
        worker_config_path=args.worker_config,
        queue_root=queue_root,
        logs_dir=logs_dir,
        template_copy_root=template_copy_root,
        preserve_local_project_copy=args.preserve_local_project_copy,
    )

    processed = 0
    idle_started = None
    while args.max_jobs is None or processed < args.max_jobs:
        running_path = _claim_one_pending(cfg.queue_root)
        if running_path is None:
            if args.once or not args.wait:
                print("status=no_pending")
                break
            if idle_started is None:
                idle_started = time.monotonic()
                print(f"status=waiting_for_pending poll_interval_sec={args.poll_interval_sec}")
            time.sleep(args.poll_interval_sec)
            continue

        idle_started = None
        job_id = _job_id_from_path(running_path)
        print(f"claimed={job_id}")
        processor.process(running_path)
        state = "done" if (cfg.queue_root / "done" / f"{job_id}.result.json").exists() else "failed"
        print(f"status={state} job_id={job_id}")
        processed += 1
        if args.once:
            break

    print(f"processed_count={processed}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local linear worker against an offline filesystem queue."
    )
    parser.add_argument("--worker-config", type=Path, default=DEFAULT_WORKER_CONFIG)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--slot-root", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--template-copy-root", type=Path, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-local-project-copy", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
