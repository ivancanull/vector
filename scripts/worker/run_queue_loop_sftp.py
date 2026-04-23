from __future__ import annotations

# Recommended Windows worker command:
#
#   cd D:\vector
#   $env:PYTHONPATH = "src"
#   .\.venv\Scripts\python.exe scripts\worker\run_queue_loop_sftp.py `
#     --worker-config configs\worker.spiral_local.yaml `
#     --server-host 10.15.89.240 `
#     --server-port 22031 `
#     --username zhanghf `
#     --remote-queue-root /home/zhanghf/projects/vector/queue
#
# One-iteration smoke test: append `--once` to the command above.
# Default SSH files are %USERPROFILE%\.ssh\id_ed25519 and
# %USERPROFILE%\.ssh\known_hosts. Override with --ssh-key/--known-hosts when needed.

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import ensure_runtime_dirs  # noqa: E402
from worker.config import load_worker_config  # noqa: E402
from worker.queue_loop import (  # noqa: E402
    ClaimedJob,
    FileLocalQueueStore,
    JobResult,
    LoggingWorkerReporter,
    LoopContext,
    SftpRemoteQueueClient,
    SftpResultUploader,
    SftpSettings,
    StaleJobPolicy,
    SubprocessJobRunner,
    WorkerLoopService,
    build_worker_loop_service,
    remote_queue_path,
    run_worker_loop,
    setup_loop_logger,
)
from worker.sftp_uploader import upload_job_bundle_via_sftp  # noqa: E402


DEFAULT_SERVER_HOST = "10.15.89.240"
DEFAULT_SERVER_PORT = 22031
DEFAULT_USERNAME = "zhanghf"
DEFAULT_REMOTE_QUEUE_ROOT = "/home/zhanghf/projects/vector/queue"
DEFAULT_STALE_RUNNING_SEC = 24 * 60 * 60
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
DEFAULT_KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts"


def _local_store(ctx: LoopContext) -> FileLocalQueueStore:
    return FileLocalQueueStore(
        queue_root=ctx.queue_root,
        worker_name=ctx.worker_name,
        stale_policy=StaleJobPolicy(ctx.stale_running_sec),
    )


def claim_one_remote_job(ctx: LoopContext) -> ClaimedJob | None:
    return SftpRemoteQueueClient(ctx.sftp, ctx.queue_root).claim_one()


def run_job_subprocess(ctx: LoopContext, job_path: Path) -> Path:
    store = _local_store(ctx)
    job_id = store.job_id_from_file(job_path)
    result = SubprocessJobRunner(
        repo_root=REPO_ROOT,
        worker_config_path=ctx.worker_config_path,
        timeout_sec=ctx.job_timeout_sec,
        local_store=store,
        run_command=subprocess.run,
    ).run(
        ClaimedJob(
            job_id=job_id,
            job_name=job_path.name,
            local_job_path=job_path,
            remote_running_path=remote_queue_path(ctx.sftp, "running", job_path.name),
        )
    )
    return result.result_json_path


def upload_result(ctx: LoopContext, result_json: Path) -> None:
    SftpResultUploader(
        queue_root=ctx.queue_root,
        settings=ctx.sftp,
        upload_fn=upload_job_bundle_via_sftp,
    ).upload(result_json)


def archive_local_result(ctx: LoopContext, result_json: Path) -> Path:
    return _local_store(ctx).archive_result(result_json)


def archive_local_running_job(ctx: LoopContext, job_path: Path) -> Path | None:
    return _local_store(ctx).archive_running_job(job_path)


def archive_remote_running_job(ctx: LoopContext, remote_running_path: str) -> None:
    SftpRemoteQueueClient(ctx.sftp, ctx.queue_root).archive_running(remote_running_path)


def process_claimed_job(ctx: LoopContext, claimed: ClaimedJob) -> None:
    store = _local_store(ctx)
    service = WorkerLoopService(
        remote_queue=SftpRemoteQueueClient(ctx.sftp, ctx.queue_root),
        local_store=store,
        runner=SubprocessJobRunner(
            repo_root=REPO_ROOT,
            worker_config_path=ctx.worker_config_path,
            timeout_sec=ctx.job_timeout_sec,
            local_store=store,
            run_command=subprocess.run,
        ),
        uploader=SftpResultUploader(
            queue_root=ctx.queue_root,
            settings=ctx.sftp,
            upload_fn=upload_job_bundle_via_sftp,
        ),
        reporter=LoggingWorkerReporter(ctx.logger),
    )
    result = service.runner.run(claimed)
    service._upload_and_archive(result.result_json_path, claimed.remote_running_path)
    store.archive_running_job(claimed.local_job_path)


def process_stale_local_running_jobs(ctx: LoopContext) -> int:
    store = _local_store(ctx)
    service = WorkerLoopService(
        remote_queue=SftpRemoteQueueClient(ctx.sftp, ctx.queue_root),
        local_store=store,
        runner=SubprocessJobRunner(
            repo_root=REPO_ROOT,
            worker_config_path=ctx.worker_config_path,
            timeout_sec=ctx.job_timeout_sec,
            local_store=store,
            run_command=subprocess.run,
        ),
        uploader=SftpResultUploader(
            queue_root=ctx.queue_root,
            settings=ctx.sftp,
            upload_fn=upload_job_bundle_via_sftp,
        ),
        reporter=LoggingWorkerReporter(ctx.logger),
    )
    stale_count = len(store.stale_running_jobs())
    service._process_stale_jobs()
    return stale_count


def process_unuploaded_local_results(ctx: LoopContext) -> int:
    store = _local_store(ctx)
    service = WorkerLoopService(
        remote_queue=SftpRemoteQueueClient(ctx.sftp, ctx.queue_root),
        local_store=store,
        runner=SubprocessJobRunner(
            repo_root=REPO_ROOT,
            worker_config_path=ctx.worker_config_path,
            timeout_sec=ctx.job_timeout_sec,
            local_store=store,
            run_command=subprocess.run,
        ),
        uploader=SftpResultUploader(
            queue_root=ctx.queue_root,
            settings=ctx.sftp,
            upload_fn=upload_job_bundle_via_sftp,
        ),
        reporter=LoggingWorkerReporter(ctx.logger),
    )
    result_count = len(store.unuploaded_results())
    service._upload_unuploaded_results()
    return result_count


def run_loop(ctx: LoopContext, once: bool) -> int:
    service = build_worker_loop_service(
        ctx=ctx,
        repo_root=REPO_ROOT,
        run_command=subprocess.run,
        upload_fn=upload_job_bundle_via_sftp,
    )
    return run_worker_loop(service, ctx, once)


def build_context(
    *,
    worker_config_path: Path,
    server_host: str,
    server_port: int,
    username: str,
    remote_queue_root: str,
    ssh_key: Path,
    known_hosts: Path,
    poll_interval_sec: int | None,
    job_timeout_sec: int | None,
    stale_running_sec: int,
) -> LoopContext:
    cfg = load_worker_config(worker_config_path)
    return LoopContext(
        worker_name=cfg.worker_name,
        worker_config_path=worker_config_path,
        queue_root=cfg.queue_root,
        logs_dir=cfg.logs_dir,
        poll_interval_sec=poll_interval_sec if poll_interval_sec is not None else cfg.poll_interval_sec,
        job_timeout_sec=job_timeout_sec if job_timeout_sec is not None else cfg.job_timeout_sec,
        stale_running_sec=stale_running_sec,
        sftp=SftpSettings(
            server_host=server_host,
            server_port=server_port,
            username=username,
            remote_queue_root=remote_queue_root,
            ssh_key=ssh_key,
            known_hosts=known_hosts,
        ),
        logger=setup_loop_logger(cfg.logs_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously fetch remote queue jobs over SFTP, run them, and upload results."
    )
    parser.add_argument(
        "--worker-config",
        type=Path,
        default=REPO_ROOT / "configs" / "worker.yaml",
        help="Path to worker config YAML.",
    )
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--remote-queue-root", default=DEFAULT_REMOTE_QUEUE_ROOT)
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=DEFAULT_SSH_KEY,
        help=f"SSH private key path. Default: {DEFAULT_SSH_KEY}",
    )
    parser.add_argument(
        "--known-hosts",
        type=Path,
        default=DEFAULT_KNOWN_HOSTS,
        help=f"known_hosts path. Default: {DEFAULT_KNOWN_HOSTS}",
    )
    parser.add_argument("--poll-interval-sec", type=int, default=None)
    parser.add_argument("--job-timeout-sec", type=int, default=None)
    parser.add_argument("--stale-running-sec", type=int, default=DEFAULT_STALE_RUNNING_SEC)
    parser.add_argument("--once", action="store_true", help="Run at most one loop iteration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = build_context(
        worker_config_path=args.worker_config,
        server_host=args.server_host,
        server_port=args.server_port,
        username=args.username,
        remote_queue_root=args.remote_queue_root,
        ssh_key=args.ssh_key,
        known_hosts=args.known_hosts,
        poll_interval_sec=args.poll_interval_sec,
        job_timeout_sec=args.job_timeout_sec,
        stale_running_sec=args.stale_running_sec,
    )
    ensure_runtime_dirs(ctx.queue_root, ctx.logs_dir)
    return run_loop(ctx, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
