from __future__ import annotations

"""Generic worker loop: fetch remote pending jobs, run locally, submit results."""

import argparse
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import ensure_runtime_dirs  # noqa: E402
from worker.config import load_templates_config, load_worker_config  # noqa: E402
from worker.logger import setup_global_logger  # noqa: E402
from worker.queue_loop import (  # noqa: E402
    SftpRemoteQueueClient,
    SftpResultUploader,
    SftpSettings,
    find_result_json_for_job,
)
from worker.template_registry import TemplateRegistry  # noqa: E402
from worker.worker_main import ClaimedJobProcessor  # noqa: E402


DEFAULT_SERVER_HOST = "10.15.89.240"
DEFAULT_SERVER_PORT = 22031
DEFAULT_USERNAME = "zhanghf"
DEFAULT_REMOTE_QUEUE_ROOT = "/home/zhanghf/projects/vector/queue"
DEFAULT_WORKER_CONFIG = REPO_ROOT / "configs" / "worker.yaml"


def _load_runtime(worker_config_path: Path, queue_root_override: Path | None):
    cfg = load_worker_config(worker_config_path)
    if queue_root_override is not None:
        cfg.queue_root = queue_root_override
        cfg.results_dir = queue_root_override / "results"
    templates_cfg = load_templates_config(cfg.templates_yaml)
    ensure_runtime_dirs(cfg.queue_root, cfg.logs_dir)
    template_registry = TemplateRegistry(templates_cfg)
    global_logger = setup_global_logger(cfg.logs_dir)
    return cfg, template_registry, global_logger


def _remove_stale_local_results(queue_root: Path, job_id: str) -> None:
    done_result = queue_root / "done" / f"{job_id}.result.json"
    failed_result = queue_root / "failed" / f"{job_id}.result.json"
    for path in (done_result, failed_result):
        if path.exists():
            path.unlink()
            print(f"removed_stale_result={path}")


def _build_sftp_settings(
    *,
    server_host: str,
    server_port: int,
    username: str,
    ssh_key: Path,
    known_hosts: Path,
    remote_queue_root: str,
) -> SftpSettings:
    return SftpSettings(
        server_host=server_host,
        server_port=server_port,
        username=username,
        remote_queue_root=remote_queue_root,
        ssh_key=ssh_key,
        known_hosts=known_hosts,
    )


def _is_retryable_sftp_connect_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True

    try:
        import paramiko  # type: ignore
    except ImportError:
        return False

    ssh_exceptions: tuple[type[BaseException], ...] = (
        paramiko.SSHException,
    )
    return isinstance(exc, ssh_exceptions)


def _format_sftp_connect_error(exc: BaseException) -> str:
    message = str(exc).replace("\n", " ").strip()
    error_class = exc.__class__.__name__
    return (
        f"status=sftp_connect_failed error_class={error_class}"
        if not message
        else f"status=sftp_connect_failed error_class={error_class} error={message}"
    )


def run(
    *,
    worker_config: Path,
    queue_root: Path | None,
    server_host: str,
    server_port: int,
    username: str,
    ssh_key: Path,
    known_hosts: Path,
    remote_queue_root: str,
    wait_when_empty: bool = True,
    poll_interval_sec: int | None = None,
    idle_status_interval_sec: int = 60,
    once: bool = False,
    max_jobs: int | None = None,
) -> int:
    cfg, template_registry, global_logger = _load_runtime(worker_config, queue_root)
    processor = ClaimedJobProcessor.default(
        cfg=cfg,
        template_registry=template_registry,
        global_logger=global_logger,
    )
    sftp_settings = _build_sftp_settings(
        server_host=server_host,
        server_port=server_port,
        username=username,
        ssh_key=ssh_key,
        known_hosts=known_hosts,
        remote_queue_root=remote_queue_root,
    )
    remote_queue = SftpRemoteQueueClient(sftp_settings, cfg.queue_root)
    uploader = SftpResultUploader(queue_root=cfg.queue_root, settings=sftp_settings)
    poll_interval = poll_interval_sec if poll_interval_sec is not None else cfg.poll_interval_sec
    completed_jobs = 0
    while max_jobs is None or completed_jobs < max_jobs:
        idle_started_at: float | None = None
        last_idle_status_at: float | None = None
        while True:
            try:
                claim = remote_queue.claim_one()
            except Exception as exc:
                if not _is_retryable_sftp_connect_error(exc):
                    raise
                print(f"{_format_sftp_connect_error(exc)} poll_interval_sec={poll_interval}")
                time.sleep(poll_interval)
                continue
            if claim is not None:
                break
            if not wait_when_empty:
                print("status=no_remote_pending")
                return 0
            now = time.monotonic()
            if idle_started_at is None:
                idle_started_at = now
                last_idle_status_at = now
                print(f"status=waiting_for_remote_pending poll_interval_sec={poll_interval}")
            elif (
                last_idle_status_at is not None
                and now - last_idle_status_at >= idle_status_interval_sec
            ):
                elapsed_sec = int(now - idle_started_at)
                print(f"status=still_waiting_for_remote_pending elapsed_sec={elapsed_sec}")
                last_idle_status_at = now
            time.sleep(poll_interval)

        print(f"claimed={claim.remote_running_path}")
        _remove_stale_local_results(cfg.queue_root, claim.job_id)
        processor.process(claim.local_job_path)
        result_json = find_result_json_for_job(cfg.queue_root, claim.job_id)
        result_state = result_json.parent.name
        upload_summary = uploader.upload(result_json)
        print(f"status={result_state}")
        print(f"result_json={result_json}")
        print(f"uploaded_count={len(upload_summary.uploaded_remote_paths)}")
        print("remote_archive=server_managed")
        completed_jobs += 1
        if once or not wait_when_empty:
            break
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously fetch remote pending jobs, run locally, and upload result bundles."
    )
    parser.add_argument("--worker-config", type=Path, default=DEFAULT_WORKER_CONFIG)
    parser.add_argument("--queue-root", type=Path, default=None)
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--remote-queue-root", default=DEFAULT_REMOTE_QUEUE_ROOT)
    parser.add_argument(
        "--wait-when-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When remote pending is empty, keep polling until a job appears.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=None,
        help="Polling interval in seconds while waiting (defaults to worker config poll_interval_sec).",
    )
    parser.add_argument(
        "--idle-status-interval-sec",
        type=int,
        default=60,
        help="How often to print an idle waiting status while no remote pending job exists.",
    )
    parser.add_argument("--once", action="store_true", help="Process one job and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        worker_config=args.worker_config,
        queue_root=args.queue_root,
        server_host=args.server_host,
        server_port=args.server_port,
        username=args.username,
        ssh_key=args.ssh_key,
        known_hosts=args.known_hosts,
        remote_queue_root=args.remote_queue_root,
        wait_when_empty=args.wait_when_empty,
        poll_interval_sec=args.poll_interval_sec,
        idle_status_interval_sec=args.idle_status_interval_sec,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
