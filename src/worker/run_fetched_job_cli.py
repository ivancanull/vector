from __future__ import annotations

"""Run one fetched job JSON with optional single retry.

This module is template-agnostic and intended for internal worker use.
"""

import json
from pathlib import Path

from common.io_utils import read_json, write_json
from common.paths import ensure_runtime_dirs
from worker.config import load_templates_config, load_worker_config
from worker.logger import setup_global_logger
from worker.template_registry import TemplateRegistry
from worker.worker_main import ClaimedJobProcessor


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
    processor = ClaimedJobProcessor.default(
        cfg=cfg,
        template_registry=template_registry,
        global_logger=global_logger,
    )
    processor.process(job_file)
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
) -> tuple[dict | None, str | None]:
    if failure_class in {"mesh_failure", "export_failure"}:
        # Generic fix: drop explicit overrides and let template defaults resolve.
        updated = dict(payload)
        updated.pop("setup_name", None)
        updated.pop("sweep_name", None)
        updated.pop("design_name", None)
        return updated, "cleared_design_setup_sweep_overrides"

    if failure_class == "invalid_job_json":
        output_name = payload.get("output_filename")
        if isinstance(output_name, str) and not _touchstone_suffix_ok(output_name):
            updated = dict(payload)
            updated["output_filename"] = f"{output_name}.s2p"
            return updated, "normalized_output_filename_suffix"

    return None, None


def run_fetched_job(job_file: Path, worker_config_path: Path, retry_on_failure: bool) -> int:
    if not job_file.exists():
        print(f"Job file not found: {job_file}")
        return 2

    try:
        payload = read_json(job_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid job JSON: {job_file} ({exc})")
        return 2

    job_id = _job_id_from_payload(payload, job_file)

    _remove_stale_result_files(job_file, job_id)
    cfg = _run_once(job_file, worker_config_path)
    done_path, failed_path = _result_paths(cfg.queue_root, job_id)
    if done_path.exists():
        result = read_json(done_path)
        print("status=done")
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
        print("status=unknown")
        print(f"message=no done/failed result found for job_id={job_id}")
        return 3

    failure_payload = read_json(failed_path)
    failure_class = str(failure_payload.get("error_message", "unknown"))
    log_rel = None
    if isinstance(failure_payload.get("artifacts"), dict):
        log_rel = failure_payload["artifacts"].get("log")
    print("status=failed")
    print(f"result_json={failed_path}")
    if log_rel:
        print(f"log={cfg.queue_root / log_rel}")
    print(f"error={failure_class}")

    if not retry_on_failure:
        return 1

    updated_payload, fix_name = _apply_retry_fix(payload, failure_class)
    if updated_payload is None or fix_name is None:
        return 1

    retry_job_file = _ensure_running_job_file(job_file, updated_payload, cfg.queue_root)
    _remove_stale_result_files(retry_job_file, job_id)
    retry_cfg = _run_once(retry_job_file, worker_config_path)
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
