from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Protocol

from common.io_utils import atomic_write_json, read_json
from common.paths import ensure_runtime_dirs
from worker.utils import as_posix_rel


@dataclass(frozen=True)
class SftpSettings:
    server_host: str
    server_port: int
    username: str
    remote_queue_root: str
    ssh_key: Path
    known_hosts: Path


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    job_name: str
    local_job_path: Path
    remote_running_path: str


@dataclass(frozen=True)
class JobResult:
    job_id: str
    result_json_path: Path
    status: str


@dataclass(frozen=True)
class UploadSummary:
    status: str
    result_json_path: Path
    uploaded_remote_paths: list[str]
    total_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class LoopContext:
    worker_name: str
    worker_config_path: Path
    queue_root: Path
    logs_dir: Path
    poll_interval_sec: int
    job_timeout_sec: int
    stale_running_sec: int
    sftp: SftpSettings
    logger: logging.Logger


class RemoteQueueClient(Protocol):
    def claim_one(self) -> ClaimedJob | None:
        ...

    def running_path_for_local_job(self, job_path: Path) -> str:
        ...


class LocalQueueStore(Protocol):
    def stale_running_jobs(self) -> list[Path]:
        ...

    def unuploaded_results(self) -> list[Path]:
        ...

    def job_id_from_file(self, job_path: Path) -> str:
        ...

    def job_id_from_result(self, result_json_path: Path) -> str | None:
        ...

    def find_result_json(self, job_id: str) -> Path | None:
        ...

    def write_failure_result(
        self, job_path: Path, error_message: str, detail: str = ""
    ) -> Path:
        ...

    def archive_result(self, result_json_path: Path) -> Path:
        ...

    def archive_running_job(self, job_path: Path) -> Path | None:
        ...


class JobRunner(Protocol):
    def run(self, claimed: ClaimedJob) -> JobResult:
        ...


class ResultUploader(Protocol):
    def upload(self, result_json_path: Path) -> UploadSummary:
        ...


class WorkerReporter(Protocol):
    def event(self, message: str) -> None:
        ...


@dataclass(frozen=True)
class StaleJobPolicy:
    stale_after_sec: int

    def is_stale(self, path: Path, now: float) -> bool:
        return now - path.stat().st_mtime >= self.stale_after_sec


class LoggingWorkerReporter:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def event(self, message: str) -> None:
        print(message)
        self.logger.info(message)


def remote_join(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part]
    if not cleaned:
        return ""
    prefix = "/" if parts[0].startswith("/") else ""
    return prefix + "/".join(cleaned)


def remote_queue_path(settings: SftpSettings, *parts: str) -> str:
    return remote_join(settings.remote_queue_root.rstrip("/"), *parts)


def setup_loop_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hfss_worker.sftp_loop")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(logs_dir / "sftp_worker_loop.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def connect_sftp(settings: SftpSettings):
    return connect_sftp_with_timeout(settings, timeout_seconds=30)


def connect_sftp_with_timeout(settings: SftpSettings, *, timeout_seconds: int):
    if not settings.ssh_key.exists():
        raise FileNotFoundError(f"SSH key not found: {settings.ssh_key}")
    if not settings.known_hosts.exists():
        raise FileNotFoundError(f"known_hosts not found: {settings.known_hosts}")

    import paramiko  # type: ignore

    ssh_client = paramiko.SSHClient()
    ssh_client.load_host_keys(str(settings.known_hosts))
    ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh_client.connect(
        hostname=settings.server_host,
        port=settings.server_port,
        username=settings.username,
        key_filename=str(settings.ssh_key),
        look_for_keys=False,
        allow_agent=False,
        timeout=timeout_seconds,
    )
    return ssh_client, ssh_client.open_sftp()


def close_sftp(ssh_client, sftp) -> None:
    try:
        sftp.close()
    finally:
        ssh_client.close()


def sftp_mkdir_p(sftp, remote_dir: str) -> None:
    parts = [part for part in remote_dir.split("/") if part]
    current = "/" if remote_dir.startswith("/") else ""
    for part in parts:
        if current in ("", "/"):
            current = f"{current}{part}" if current == "/" else part
        else:
            current = f"{current}/{part}"
        try:
            sftp.stat(current)
        except Exception:
            sftp.mkdir(current)


def find_result_json_for_job(queue_root: Path, job_id: str) -> Path:
    done_path = queue_root / "done" / f"{job_id}.result.json"
    failed_path = queue_root / "failed" / f"{job_id}.result.json"
    if done_path.exists() and failed_path.exists():
        return done_path if done_path.stat().st_mtime >= failed_path.stat().st_mtime else failed_path
    if done_path.exists():
        return done_path
    if failed_path.exists():
        return failed_path
    raise FileNotFoundError(f"No done/failed result JSON found for job_id={job_id}")


def _relative_to_queue(queue_root: Path, path: Path) -> Path:
    return path.resolve().relative_to(queue_root.resolve())


def _collect_upload_bundle(queue_root: Path, result_json_path: Path) -> list[tuple[Path, Path]]:
    if not result_json_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {result_json_path}")

    payload = read_json(result_json_path)
    files: list[Path] = [result_json_path]
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("touchstone", "csv", "log"):
            rel_path = artifacts.get(key)
            if isinstance(rel_path, str) and rel_path:
                files.append(queue_root / rel_path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)

    bundle: list[tuple[Path, Path]] = []
    for local_path in deduped:
        if not local_path.exists():
            raise FileNotFoundError(f"Referenced artifact missing: {local_path}")
        rel_path = _relative_to_queue(queue_root, local_path)
        bundle.append((local_path, rel_path))
    return bundle


def safe_remote_rename(sftp, source: str, destination: str) -> bool:
    try:
        sftp_mkdir_p(sftp, destination.rsplit("/", 1)[0])
        sftp.rename(source, destination)
        return True
    except Exception:
        return False


class SftpRemoteQueueClient:
    def __init__(self, settings: SftpSettings, local_queue_root: Path) -> None:
        self.settings = settings
        self.local_queue_root = local_queue_root

    def claim_one(self) -> ClaimedJob | None:
        ssh_client, sftp = connect_sftp(self.settings)
        try:
            pending_dir = remote_queue_path(self.settings, "pending")
            running_dir = remote_queue_path(self.settings, "running")
            sftp_mkdir_p(sftp, pending_dir)
            sftp_mkdir_p(sftp, running_dir)

            for name in sorted(sftp.listdir(pending_dir)):
                if not name.endswith(".json"):
                    continue
                remote_pending = remote_join(pending_dir, name)
                remote_running = remote_join(running_dir, name)
                # Remote rename is the cross-machine claim lock.
                if not safe_remote_rename(sftp, remote_pending, remote_running):
                    continue

                local_running = self.local_queue_root / "running" / name
                local_running.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_running, str(local_running))
                return ClaimedJob(
                    job_id=FileLocalQueueStore.job_id_from_path(local_running),
                    job_name=name,
                    local_job_path=local_running,
                    remote_running_path=remote_running,
                )
        finally:
            close_sftp(ssh_client, sftp)
        return None

    def running_path_for_local_job(self, job_path: Path) -> str:
        return remote_queue_path(self.settings, "running", job_path.name)


class FileLocalQueueStore:
    def __init__(
        self,
        *,
        queue_root: Path,
        worker_name: str,
        stale_policy: StaleJobPolicy,
    ) -> None:
        self.queue_root = queue_root
        self.worker_name = worker_name
        self.stale_policy = stale_policy

    @staticmethod
    def job_id_from_path(path: Path) -> str:
        try:
            payload = read_json(path)
            job_id = payload.get("job_id")
            if isinstance(job_id, str) and job_id:
                return job_id
        except Exception:
            pass
        return path.stem.replace(".result", "")

    def stale_running_jobs(self) -> list[Path]:
        now = time.time()
        running_dir = self.queue_root / "running"
        running_dir.mkdir(parents=True, exist_ok=True)
        return [
            path
            for path in sorted(running_dir.glob("*.json"))
            if self.stale_policy.is_stale(path, now)
        ]

    def unuploaded_results(self) -> list[Path]:
        results: list[Path] = []
        for state in ("done", "failed"):
            result_dir = self.queue_root / state
            result_dir.mkdir(parents=True, exist_ok=True)
            results.extend(sorted(result_dir.glob("*.result.json")))
        return results

    def job_id_from_file(self, job_path: Path) -> str:
        return self.job_id_from_path(job_path)

    def job_id_from_result(self, result_json_path: Path) -> str | None:
        try:
            job_id = read_json(result_json_path).get("job_id")
        except Exception:
            return None
        return job_id if isinstance(job_id, str) and job_id else None

    def find_result_json(self, job_id: str) -> Path | None:
        for state in ("done", "failed"):
            path = self.queue_root / state / f"{job_id}.result.json"
            if path.exists():
                return path
        return None

    def write_failure_result(
        self, job_path: Path, error_message: str, detail: str = ""
    ) -> Path:
        job_id = self.job_id_from_file(job_path)
        result_path = self.queue_root / "failed" / f"{job_id}.result.json"
        if result_path.exists():
            return result_path

        results_dir = self.queue_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / f"{job_id}.log"
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"{timestamp} {error_message}: {detail}\n")

        payload = {
            "job_id": job_id,
            "status": "failed",
            "worker_name": self.worker_name,
            "start_time": timestamp,
            "finish_time": timestamp,
            "metrics": None,
            "artifacts": {
                "touchstone": None,
                "csv": None,
                "log": as_posix_rel(log_path, self.queue_root),
            },
            "error_message": error_message,
        }
        # Timeout/stale failures follow the normal result contract so server
        # watchers parse them like HFSS-generated failures.
        atomic_write_json(result_path, payload)
        return result_path

    def archive_result(self, result_json_path: Path) -> Path:
        state = result_json_path.parent.name
        target = self.queue_root / "archive" / state / result_json_path.name
        return _archive_unique(result_json_path, target)

    def archive_running_job(self, job_path: Path) -> Path | None:
        if not job_path.exists():
            return None
        target = self.queue_root / "archive" / "running" / job_path.name
        return _archive_unique(job_path, target)


def _archive_unique(source: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        source.replace(archive_path)
        return archive_path

    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    target = archive_path.with_name(f"{archive_path.stem}.{suffix}{archive_path.suffix}")
    source.replace(target)
    return target


class SubprocessJobRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        worker_config_path: Path,
        timeout_sec: int,
        local_store: LocalQueueStore,
        run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.repo_root = repo_root
        self.worker_config_path = worker_config_path
        self.timeout_sec = timeout_sec
        self.local_store = local_store
        self.run_command = run_command

    def run(self, claimed: ClaimedJob) -> JobResult:
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "worker" / "run_fetched_spiral_job.py"),
            "--job-file",
            str(claimed.local_job_path),
            "--worker-config",
            str(self.worker_config_path),
            "--no-retry-on-failure",
        ]
        try:
            completed = self.run_command(
                command,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            detail = f"timed out after {self.timeout_sec}s"
            if exc.stdout:
                detail += f"\nstdout:\n{exc.stdout}"
            if exc.stderr:
                detail += f"\nstderr:\n{exc.stderr}"
            result_path = self.local_store.write_failure_result(
                claimed.local_job_path,
                error_message="job_timeout",
                detail=detail,
            )
            return JobResult(claimed.job_id, result_path, "failed")

        result_path = self.local_store.find_result_json(claimed.job_id)
        if result_path is None:
            result_path = self.local_store.write_failure_result(
                claimed.local_job_path,
                error_message="missing_result_json",
                detail=f"subprocess returncode={completed.returncode}",
            )

        status = "done" if result_path.parent.name == "done" else "failed"
        return JobResult(claimed.job_id, result_path, status)


class SftpResultUploader:
    def __init__(
        self,
        *,
        queue_root: Path,
        settings: SftpSettings,
        timeout_seconds: int = 30,
    ) -> None:
        self.queue_root = queue_root
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def upload(self, result_json_path: Path) -> UploadSummary:
        if not self.settings.ssh_key.exists():
            raise FileNotFoundError(f"SSH key not found: {self.settings.ssh_key}")
        if not self.settings.known_hosts.exists():
            raise FileNotFoundError(f"known_hosts not found: {self.settings.known_hosts}")

        bundle = _collect_upload_bundle(self.queue_root, result_json_path)
        start = time.monotonic()
        ssh_client, sftp = connect_sftp_with_timeout(
            self.settings,
            timeout_seconds=self.timeout_seconds,
        )
        uploaded_remote_paths: list[str] = []
        total_bytes = 0
        try:
            remote_root = self.settings.remote_queue_root.rstrip("/")
            for local_path, rel_path in bundle:
                rel_posix = rel_path.as_posix()
                remote_path = f"{remote_root}/{rel_posix}"
                remote_parent = remote_path.rsplit("/", 1)[0]
                sftp_mkdir_p(sftp, remote_parent)
                sftp.put(str(local_path), remote_path)
                uploaded_remote_paths.append(remote_path)
                total_bytes += local_path.stat().st_size
        finally:
            close_sftp(ssh_client, sftp)

        elapsed = time.monotonic() - start
        return UploadSummary(
            status="uploaded",
            result_json_path=result_json_path,
            uploaded_remote_paths=uploaded_remote_paths,
            total_bytes=total_bytes,
            elapsed_seconds=elapsed,
        )


class WorkerLoopService:
    def __init__(
        self,
        *,
        remote_queue: RemoteQueueClient,
        local_store: LocalQueueStore,
        runner: JobRunner,
        uploader: ResultUploader,
        reporter: WorkerReporter,
    ) -> None:
        self.remote_queue = remote_queue
        self.local_store = local_store
        self.runner = runner
        self.uploader = uploader
        self.reporter = reporter

    def run_once(self) -> bool:
        if self._process_stale_jobs():
            return True

        if self._upload_unuploaded_results():
            return True

        claimed = self.remote_queue.claim_one()
        if claimed is None:
            self.reporter.event("idle")
            return False

        self.reporter.event(f"claimed={claimed.job_name}")
        self.reporter.event(f"started={claimed.job_id}")
        result = self.runner.run(claimed)
        terminal_event = "completed" if result.status == "done" else "failed"
        self.reporter.event(f"{terminal_event}={result.job_id}")
        self._upload_and_archive(result.result_json_path, claimed.remote_running_path)
        self.local_store.archive_running_job(claimed.local_job_path)
        self.reporter.event(f"archived_running={claimed.local_job_path}")
        return True

    def _process_stale_jobs(self) -> bool:
        processed = False
        for job_path in self.local_store.stale_running_jobs():
            job_id = self.local_store.job_id_from_file(job_path)
            self.reporter.event(f"stale={job_id}")
            result_path = self.local_store.write_failure_result(
                job_path,
                error_message="stale_running_job",
                detail="running file exceeded stale threshold",
            )
            self._upload_and_archive(
                result_path,
                self.remote_queue.running_path_for_local_job(job_path),
            )
            self.local_store.archive_running_job(job_path)
            self.reporter.event(f"archived_running={job_path}")
            processed = True
        return processed

    def _upload_unuploaded_results(self) -> bool:
        processed = False
        for result_path in self.local_store.unuploaded_results():
            job_id = self.local_store.job_id_from_result(result_path)
            remote_running = (
                self.remote_queue.running_path_for_local_job(Path(f"{job_id}.json"))
                if job_id
                else ""
            )
            self._upload_and_archive(result_path, remote_running)
            processed = True
        return processed

    def _upload_and_archive(self, result_path: Path, remote_running_path: str) -> None:
        self.uploader.upload(result_path)
        self.reporter.event(f"uploaded={result_path}")
        if remote_running_path:
            self.reporter.event(f"remote_archive=server_managed:{remote_running_path}")
        archived = self.local_store.archive_result(result_path)
        self.reporter.event(f"archived_result={archived}")


def build_worker_loop_service(
    *,
    ctx: LoopContext,
    repo_root: Path,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> WorkerLoopService:
    local_store = FileLocalQueueStore(
        queue_root=ctx.queue_root,
        worker_name=ctx.worker_name,
        stale_policy=StaleJobPolicy(ctx.stale_running_sec),
    )
    return WorkerLoopService(
        remote_queue=SftpRemoteQueueClient(ctx.sftp, ctx.queue_root),
        local_store=local_store,
        runner=SubprocessJobRunner(
            repo_root=repo_root,
            worker_config_path=ctx.worker_config_path,
            timeout_sec=ctx.job_timeout_sec,
            local_store=local_store,
            run_command=run_command,
        ),
        uploader=SftpResultUploader(
            queue_root=ctx.queue_root,
            settings=ctx.sftp,
        ),
        reporter=LoggingWorkerReporter(ctx.logger),
    )


def run_worker_loop(service: WorkerLoopService, ctx: LoopContext, once: bool) -> int:
    ensure_runtime_dirs(ctx.queue_root, ctx.logs_dir)
    while True:
        try:
            service.run_once()
        except KeyboardInterrupt:
            ctx.logger.info("sftp worker loop interrupted")
            return 130
        except Exception:
            ctx.logger.exception("sftp worker loop iteration failed")
            if once:
                return 1
            time.sleep(ctx.poll_interval_sec)

        if once:
            return 0
