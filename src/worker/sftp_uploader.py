from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from common.io_utils import read_json


@dataclass(frozen=True)
class UploadSummary:
    status: str
    result_json_path: Path
    uploaded_remote_paths: list[str]
    total_bytes: int
    elapsed_seconds: float


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


def _sftp_mkdir_p(sftp, remote_dir: str) -> None:
    if not remote_dir:
        return

    parts = [p for p in remote_dir.split("/") if p]
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


def upload_job_bundle_via_sftp(
    *,
    queue_root: Path,
    server_host: str,
    server_port: int,
    username: str,
    private_key_path: Path,
    known_hosts_path: Path,
    remote_queue_root: str,
    result_json_path: Path,
    timeout_seconds: int = 30,
) -> UploadSummary:
    if not private_key_path.exists():
        raise FileNotFoundError(f"SSH key not found: {private_key_path}")
    if not known_hosts_path.exists():
        raise FileNotFoundError(f"known_hosts not found: {known_hosts_path}")

    bundle = _collect_upload_bundle(queue_root, result_json_path)
    start = time.monotonic()

    import paramiko  # type: ignore

    ssh_client = paramiko.SSHClient()
    ssh_client.load_host_keys(str(known_hosts_path))
    ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy())

    uploaded_remote_paths: list[str] = []
    total_bytes = 0
    sftp = None
    try:
        ssh_client.connect(
            hostname=server_host,
            port=server_port,
            username=username,
            key_filename=str(private_key_path),
            look_for_keys=False,
            allow_agent=False,
            timeout=timeout_seconds,
        )
        sftp = ssh_client.open_sftp()
        remote_root = remote_queue_root.rstrip("/")
        for local_path, rel_path in bundle:
            rel_posix = rel_path.as_posix()
            remote_path = f"{remote_root}/{rel_posix}"
            remote_parent = remote_path.rsplit("/", 1)[0]
            _sftp_mkdir_p(sftp, remote_parent)
            sftp.put(str(local_path), remote_path)
            uploaded_remote_paths.append(remote_path)
            total_bytes += local_path.stat().st_size
    finally:
        if sftp is not None:
            sftp.close()
        ssh_client.close()

    elapsed = time.monotonic() - start
    return UploadSummary(
        status="uploaded",
        result_json_path=result_json_path,
        uploaded_remote_paths=uploaded_remote_paths,
        total_bytes=total_bytes,
        elapsed_seconds=elapsed,
    )
