from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.io_utils import read_json, write_json  # noqa: E402
from common.paths import ensure_runtime_dirs  # noqa: E402
from worker.config import load_templates_config, load_worker_config  # noqa: E402
from worker.logger import setup_global_logger  # noqa: E402
from worker.template_registry import TemplateRegistry  # noqa: E402
from worker.worker_main import process_claimed_job  # noqa: E402


LOCAL_FALLBACK_WORKER_CONFIG = REPO_ROOT / "configs" / "worker.spiral_local.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one already-fetched local job JSON and optionally auto-retry once."
    )
    parser.add_argument(
        "--job-file",
        type=Path,
        required=True,
        help="Local fetched job JSON path (typically under queue/running).",
    )
    parser.add_argument(
        "--worker-config",
        type=Path,
        default=LOCAL_FALLBACK_WORKER_CONFIG,
        help="Path to worker config YAML.",
    )
    parser.add_argument(
        "--retry-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-apply one deterministic fix and retry once.",
    )
    return parser.parse_args()


def _touchstone_suffix_ok(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix.startswith(".s") and suffix.endswith("p")


def _job_id_from_payload(payload: dict, job_file: Path) -> str:
    value = payload.get("job_id")
    if isinstance(value, str) and value:
        return value
    return job_file.stem


def _result_paths(queue_root: Path, job_id: str) -> tuple[Path, Path]:
    return (
        queue_root / "done" / f"{job_id}.result.json",
        queue_root / "failed" / f"{job_id}.result.json",
    )


def _queue_root_from_job_file(job_file: Path) -> Path | None:
    if job_file.parent.name != "running":
        return None
    return job_file.parent.parent


def _remove_stale_result_files(job_file: Path, job_id: str) -> None:
    queue_root = _queue_root_from_job_file(job_file)
    if queue_root is None:
        return
    done_path, failed_path = _result_paths(queue_root, job_id)
    for path in (done_path, failed_path):
        if path.exists():
            path.unlink()
            print(f"removed_stale_result={path}")


def _run_once(job_file: Path, worker_config_path: Path):
    cfg = load_worker_config(worker_config_path)
    templates_cfg = load_templates_config(cfg.templates_yaml)
    ensure_runtime_dirs(cfg.queue_root, cfg.logs_dir)
    template_registry = TemplateRegistry(templates_cfg)
    global_logger = setup_global_logger(cfg.logs_dir)
    process_claimed_job(job_file, cfg, template_registry, global_logger)
    return cfg


def _ensure_running_job_file(job_file: Path, payload: dict, queue_root: Path) -> Path:
    running_dir = queue_root / "running"
    running_dir.mkdir(parents=True, exist_ok=True)
    target = running_dir / job_file.name
    write_json(target, payload)
    return target


def _apply_retry_fix(
    payload: dict,
    failure_class: str,
    worker_config_path: Path,
) -> tuple[dict | None, Path, str | None]:
    if failure_class == "missing_template_file":
        return payload, LOCAL_FALLBACK_WORKER_CONFIG, "switched_worker_config_to_local_template"

    if failure_class in {"mesh_failure", "export_failure"}:
        updated = dict(payload)
        updated["setup_name"] = "Setup1"
        updated["sweep_name"] = "Sweep"
        return updated, worker_config_path, "forced_setup_sweep_to_Setup1_Sweep"

    if failure_class == "invalid_job_json":
        output_name = payload.get("output_filename")
        if isinstance(output_name, str) and not _touchstone_suffix_ok(output_name):
            updated = dict(payload)
            updated["output_filename"] = f"{output_name}.s2p"
            return updated, worker_config_path, "normalized_output_filename_suffix"

    return None, worker_config_path, None


def run_fetched_job(job_file: Path, worker_config_path: Path, retry_on_failure: bool) -> int:
    if not job_file.exists():
        print(f"Job file not found: {job_file}", file=sys.stderr)
        return 2

    try:
        payload = read_json(job_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid job JSON: {job_file} ({exc})", file=sys.stderr)
        return 2

    job_id = _job_id_from_payload(payload, job_file)

    _remove_stale_result_files(job_file, job_id)
    cfg = _run_once(job_file, worker_config_path)
    done_path, failed_path = _result_paths(cfg.queue_root, job_id)
    if done_path.exists():
        result = read_json(done_path)
        print(f"status=done")
        print(f"result_json={done_path}")
        if isinstance(result.get("artifacts"), dict):
            artifacts = result["artifacts"]
            touchstone = artifacts.get("touchstone")
            log_path = artifacts.get("log")
            if touchstone:
                print(f"touchstone={cfg.queue_root / touchstone}")
            if log_path:
                print(f"log={cfg.queue_root / log_path}")
        return 0

    if not failed_path.exists():
        print(f"status=unknown", file=sys.stderr)
        print(f"message=no done/failed result found for job_id={job_id}", file=sys.stderr)
        return 3

    failure_payload = read_json(failed_path)
    failure_class = str(failure_payload.get("error_message", "unknown"))
    log_rel = None
    if isinstance(failure_payload.get("artifacts"), dict):
        log_rel = failure_payload["artifacts"].get("log")
    print(f"status=failed")
    print(f"result_json={failed_path}")
    if log_rel:
        print(f"log={cfg.queue_root / log_rel}")
    print(f"error={failure_class}")

    if not retry_on_failure:
        return 1

    updated_payload, retry_worker_config_path, fix_name = _apply_retry_fix(
        payload, failure_class, worker_config_path
    )
    if updated_payload is None or fix_name is None:
        return 1

    retry_job_file = _ensure_running_job_file(job_file, updated_payload, cfg.queue_root)
    _remove_stale_result_files(retry_job_file, job_id)
    retry_cfg = _run_once(retry_job_file, retry_worker_config_path)
    retry_done_path, retry_failed_path = _result_paths(retry_cfg.queue_root, job_id)
    print(f"retry_fix={fix_name}")
    if retry_done_path.exists():
        retry_result = read_json(retry_done_path)
        print("status=done_after_retry")
        print(f"result_json={retry_done_path}")
        if isinstance(retry_result.get("artifacts"), dict):
            artifacts = retry_result["artifacts"]
            touchstone = artifacts.get("touchstone")
            log_path = artifacts.get("log")
            if touchstone:
                print(f"touchstone={retry_cfg.queue_root / touchstone}")
            if log_path:
                print(f"log={retry_cfg.queue_root / log_path}")
        return 0

    if retry_failed_path.exists():
        retry_failure = read_json(retry_failed_path)
        retry_error = str(retry_failure.get("error_message", "unknown"))
        print("status=failed_after_retry")
        print(f"result_json={retry_failed_path}")
        print(f"error={retry_error}")
        if isinstance(retry_failure.get("artifacts"), dict):
            retry_log = retry_failure["artifacts"].get("log")
            if retry_log:
                print(f"log={retry_cfg.queue_root / retry_log}")
    return 1


def main() -> int:
    args = parse_args()
    return run_fetched_job(args.job_file, args.worker_config, args.retry_on_failure)


if __name__ == "__main__":
    raise SystemExit(main())
