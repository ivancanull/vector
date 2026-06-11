from __future__ import annotations

"""Run fixed new-spiral cases through the same queue job path used by BCLF."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.io_utils import read_json, write_json  # noqa: E402
from common.paths import ensure_runtime_dirs  # noqa: E402
from scripts.server.submit_spiral_batch import parse_batch_payload  # noqa: E402
from server.queue_manager import QueueManager  # noqa: E402
from worker.config import load_templates_config, load_worker_config  # noqa: E402
from worker.logger import setup_global_logger  # noqa: E402
from worker.template_registry import TemplateRegistry  # noqa: E402
from worker.worker_main import ClaimedJobProcessor  # noqa: E402


DEFAULT_WORKER_CONFIG = REPO_ROOT / "configs" / "worker.yaml"
DEFAULT_TEMPLATE_NAME = "spiral_inductor_t3_template_v1"
DEFAULT_DESIGN_NAME = "SpiralInductorT3"
DEFAULT_SETUP_NAME = "Setup1"
DEFAULT_SWEEP_NAME = "Sweep"
DEFAULT_BATCH_ID = "spiral_t3_validation"
DEFAULT_QUEUE_ROOT = REPO_ROOT / ".runtime" / "local_spiral_t3_parallel_validation"
DEFAULT_SLOT_ROOT = REPO_ROOT / ".runtime" / "local_spiral_t3_parallel_slots"

FIXED_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "nominal",
        "variables": {"Din": "60um", "W": "15um", "S": "1.5um", "H": "2um"},
        "output_filename": "spiral_t3_nominal.s2p",
    },
    {
        "case_id": "min_geometry",
        "variables": {"Din": "50um", "W": "12um", "S": "1um", "H": "2um"},
        "output_filename": "spiral_t3_min_geometry.s2p",
    },
    {
        "case_id": "max_geometry",
        "variables": {"Din": "70um", "W": "18um", "S": "2um", "H": "2um"},
        "output_filename": "spiral_t3_max_geometry.s2p",
    },
    {
        "case_id": "wide_low_spacing",
        "variables": {"Din": "70um", "W": "12um", "S": "1um", "H": "2um"},
        "output_filename": "spiral_t3_wide_low_spacing.s2p",
    },
    {
        "case_id": "compact_high_spacing",
        "variables": {"Din": "50um", "W": "18um", "S": "2um", "H": "2um"},
        "output_filename": "spiral_t3_compact_high_spacing.s2p",
    },
)


@dataclass(frozen=True)
class LocalParallelSlot:
    index: int
    name: str
    root: Path
    worker_config: Path


def build_fixed_batch_payload(
    *,
    batch_id: str = DEFAULT_BATCH_ID,
    template_name: str = DEFAULT_TEMPLATE_NAME,
    design_name: str = DEFAULT_DESIGN_NAME,
    setup_name: str = DEFAULT_SETUP_NAME,
    sweep_name: str = DEFAULT_SWEEP_NAME,
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "template_name": template_name,
        "design_name": design_name,
        "setup_name": setup_name,
        "sweep_name": sweep_name,
        "metrics_to_extract": ["max_s11_db", "min_s21_db"],
        "artifact_options": {"export_touchstone": True, "export_csv": True},
        "retry_count": 0,
        "cases": list(FIXED_CASES),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit and run fixed new-spiral cases through the local worker queue path."
    )
    parser.add_argument("--worker-config", type=Path, default=DEFAULT_WORKER_CONFIG)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--template-name", default=DEFAULT_TEMPLATE_NAME)
    parser.add_argument("--design-name", default=DEFAULT_DESIGN_NAME)
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument(
        "--parallel-slots",
        type=int,
        default=1,
        help="Run local validation jobs with this many isolated worker slots.",
    )
    parser.add_argument(
        "--slot-root",
        type=Path,
        default=DEFAULT_SLOT_ROOT,
        help="Root directory for generated local parallel slot configs/logs/AEDT copies.",
    )
    parser.add_argument(
        "--local-slot-runner",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the batch JSON and do not touch the queue.")
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Emit pending jobs but do not run them locally.",
    )
    return parser.parse_args(argv)


def _load_processor(worker_config_path: Path, queue_root_override: Path | None):
    cfg = load_worker_config(worker_config_path)
    if queue_root_override is not None:
        cfg.queue_root = queue_root_override
        cfg.results_dir = queue_root_override / "results"
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


def build_local_slot_worker_config_payload(
    base_cfg,
    *,
    slot_index: int,
    slot_root: Path,
) -> dict[str, Any]:
    slot_name = f"slot_{slot_index:02d}"
    root = slot_root / slot_name
    payload = base_cfg.model_dump(mode="json")
    payload.update(
        {
            "worker_name": f"{base_cfg.worker_name}_local_{slot_name}",
            "logs_dir": str(root / "logs"),
            "new_desktop": True,
            "close_desktop_after_job": True,
            "copy_template_per_job": True,
            "template_copy_root": str(root / "aedt_copies"),
        }
    )
    return payload


def write_local_slot_worker_configs(
    *,
    base_cfg,
    slot_root: Path,
    slots: int,
) -> list[LocalParallelSlot]:
    emitted: list[LocalParallelSlot] = []
    slot_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, slots + 1):
        name = f"slot_{index:02d}"
        root = slot_root / name
        root.mkdir(parents=True, exist_ok=True)
        payload = build_local_slot_worker_config_payload(
            base_cfg,
            slot_index=index,
            slot_root=slot_root,
        )
        config_path = root / "worker.generated.yaml"
        with config_path.open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        emitted.append(LocalParallelSlot(index=index, name=name, root=root, worker_config=config_path))
    return emitted


def _remove_stale_results(queue_root: Path, job_id: str) -> None:
    for state in ("done", "failed"):
        path = queue_root / state / f"{job_id}.result.json"
        if path.exists():
            path.unlink()
            print(f"removed_stale_result={path}")


def _remove_stale_running_job(queue_root: Path, job_id: str) -> None:
    running_path = queue_root / "running" / f"{job_id}.json"
    if running_path.exists():
        running_path.unlink()
        print(f"removed_stale_running={running_path}")


def _move_pending_to_running(pending_path: Path, queue_root: Path) -> Path:
    running_path = queue_root / "running" / pending_path.name
    running_path.parent.mkdir(parents=True, exist_ok=True)
    if running_path.exists():
        running_path.unlink()
    pending_path.replace(running_path)
    return running_path


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
        except FileExistsError:
            continue
        except OSError:
            continue
        try:
            if running_path.exists():
                continue
            pending_path.replace(running_path)
            return running_path
        except FileNotFoundError:
            continue
        except OSError:
            continue
        finally:
            try:
                claim_lock.unlink()
            except FileNotFoundError:
                pass
    return None


def _result_for_job(queue_root: Path, job_id: str) -> tuple[str, Path | None, dict[str, Any] | None]:
    for state in ("done", "failed"):
        path = queue_root / state / f"{job_id}.result.json"
        if path.exists():
            return state, path, read_json(path)
    return "missing", None, None


def _summarize_job(queue_root: Path, job_id: str) -> dict[str, Any]:
    state, result_path, result_payload = _result_for_job(queue_root, job_id)
    artifacts = result_payload.get("artifacts", {}) if isinstance(result_payload, dict) else {}
    return {
        "job_id": job_id,
        "state": state,
        "worker_name": result_payload.get("worker_name") if isinstance(result_payload, dict) else None,
        "result_json": str(result_path) if result_path is not None else None,
        "touchstone": artifacts.get("touchstone") if isinstance(artifacts, dict) else None,
        "error_message": result_payload.get("error_message") if isinstance(result_payload, dict) else None,
    }


def _write_summary(
    *,
    cfg,
    batch_id: str,
    template_name: str,
    jobs: list[dict[str, Any]],
    parallel_slots: int,
    slot_configs: list[LocalParallelSlot] | None = None,
) -> Path:
    summary_path = cfg.queue_root / "results" / f"{batch_id}_summary.json"
    write_json(
        summary_path,
        {
            "batch_id": batch_id,
            "template_name": template_name,
            "queue_root": str(cfg.queue_root),
            "parallel_slots": parallel_slots,
            "slot_configs": [str(slot.worker_config) for slot in (slot_configs or [])],
            "jobs": jobs,
        },
    )
    print(f"summary={summary_path}")
    return summary_path


def process_pending_jobs_in_slot(cfg, processor) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    while True:
        running_path = _claim_one_pending(cfg.queue_root)
        if running_path is None:
            break
        job_payload = read_json(running_path)
        job_id = str(job_payload["job_id"])
        print(f"slot={cfg.worker_name} claimed={job_id}")
        processor.process(running_path)
        summary = _summarize_job(cfg.queue_root, job_id)
        summaries.append(summary)
        print(f"slot={cfg.worker_name} status={summary['state']} job_id={job_id}")
    print(f"slot={cfg.worker_name} processed_count={len(summaries)}")
    return summaries


def run_local_slot(args: argparse.Namespace) -> int:
    cfg, processor = _load_processor(args.worker_config, args.queue_root)
    process_pending_jobs_in_slot(cfg, processor)
    return 0


def _run_parallel_slots(
    *,
    cfg,
    worker_config: Path,
    slot_root: Path,
    parallel_slots: int,
    expected_job_ids: list[str],
    batch_id: str,
    template_name: str,
) -> int:
    slots = write_local_slot_worker_configs(
        base_cfg=cfg,
        slot_root=slot_root,
        slots=parallel_slots,
    )
    processes: list[subprocess.Popen] = []
    try:
        for slot in slots:
            command = [
                sys.executable,
                str(Path(__file__)),
                "--local-slot-runner",
                "--worker-config",
                str(slot.worker_config),
                "--queue-root",
                str(cfg.queue_root),
            ]
            print(f"slot={slot.name} worker_config={slot.worker_config}")
            print(f"launch={' '.join(command)}")
            processes.append(subprocess.Popen(command, cwd=REPO_ROOT))
        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        return 130

    summaries = [_summarize_job(cfg.queue_root, job_id) for job_id in expected_job_ids]
    _write_summary(
        cfg=cfg,
        batch_id=batch_id,
        template_name=template_name,
        jobs=summaries,
        parallel_slots=parallel_slots,
        slot_configs=slots,
    )
    failed = [row for row in summaries if row["state"] != "done"]
    return 0 if not failed and all(code == 0 for code in return_codes) else 1


def run(args: argparse.Namespace) -> int:
    if getattr(args, "local_slot_runner", False):
        return run_local_slot(args)
    parallel_slots = int(getattr(args, "parallel_slots", 1))
    if parallel_slots <= 0:
        print("invalid_args=parallel-slots must be positive", file=sys.stderr)
        return 2

    payload = build_fixed_batch_payload(
        batch_id=args.batch_id,
        template_name=args.template_name,
        design_name=args.design_name,
        setup_name=args.setup_name,
        sweep_name=args.sweep_name,
    )
    # Reuse submit_spiral_batch validation before any queue mutation.
    batch = parse_batch_payload(payload)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    cfg, processor = _load_processor(args.worker_config, args.queue_root)
    manager = QueueManager(cfg.queue_root)
    emitted = manager.enqueue_batch(batch)
    print(f"submitted_count={len(emitted)}")

    if args.submit_only:
        for path in emitted:
            print(f"pending={path}")
        return 0

    expected_job_ids: list[str] = []
    for pending_path in emitted:
        job_id = str(read_json(pending_path)["job_id"])
        expected_job_ids.append(job_id)
        _remove_stale_results(cfg.queue_root, job_id)
        _remove_stale_running_job(cfg.queue_root, job_id)

    if parallel_slots > 1:
        return _run_parallel_slots(
            cfg=cfg,
            worker_config=args.worker_config,
            slot_root=args.slot_root,
            parallel_slots=parallel_slots,
            expected_job_ids=expected_job_ids,
            batch_id=args.batch_id,
            template_name=args.template_name,
        )

    summaries: list[dict[str, Any]] = []
    for pending_path in emitted:
        job_payload = read_json(pending_path)
        job_id = str(job_payload["job_id"])
        running_path = _move_pending_to_running(pending_path, cfg.queue_root)
        print(f"running={running_path}")
        processor.process(running_path)
        summary = _summarize_job(cfg.queue_root, job_id)
        summaries.append(summary)
        print(f"status={summary['state']} job_id={job_id}")

    _write_summary(
        cfg=cfg,
        batch_id=args.batch_id,
        template_name=args.template_name,
        jobs=summaries,
        parallel_slots=1,
    )
    failed = [row for row in summaries if row["state"] != "done"]
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
