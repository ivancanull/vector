from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from worker.queue_loop import SftpResultUploader, SftpSettings  # noqa: E402


DEFAULT_QUEUE_ROOT = REPO_ROOT / "queue"
DEFAULT_SERVER_HOST = "10.15.89.240"
DEFAULT_SERVER_PORT = 22031
DEFAULT_USERNAME = "zhanghf"
DEFAULT_REMOTE_QUEUE_ROOT = "/home/zhanghf/projects/vector/queue"
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
DEFAULT_KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts"


def _find_latest_result_json(queue_root: Path) -> Path | None:
    candidates = list((queue_root / "done").glob("*.result.json"))
    candidates.extend((queue_root / "failed").glob("*.result.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload latest queue job result bundle to server via SFTP."
    )
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
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
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Explicit result JSON path to upload (otherwise latest done/failed result is used).",
    )
    return parser.parse_args()


def run_upload(
    *,
    queue_root: Path,
    server_host: str,
    server_port: int,
    username: str,
    remote_queue_root: str,
    ssh_key: Path,
    known_hosts: Path,
    result_json: Path | None,
) -> int:
    selected = result_json if result_json is not None else _find_latest_result_json(queue_root)
    if selected is None:
        print("No result JSON found in queue/done or queue/failed.", file=sys.stderr)
        return 2

    uploader = SftpResultUploader(
        queue_root=queue_root,
        settings=SftpSettings(
            server_host=server_host,
            server_port=server_port,
            username=username,
            remote_queue_root=remote_queue_root,
            ssh_key=ssh_key,
            known_hosts=known_hosts,
        ),
    )
    summary = uploader.upload(selected)
    print(f"selected_result_json={summary.result_json_path}")
    for remote in summary.uploaded_remote_paths:
        print(f"uploaded={remote}")
    print(f"total_bytes={summary.total_bytes}")
    print(f"status={summary.status}")
    return 0


def main() -> int:
    args = parse_args()
    return run_upload(
        queue_root=args.queue_root,
        server_host=args.server_host,
        server_port=args.server_port,
        username=args.username,
        remote_queue_root=args.remote_queue_root,
        ssh_key=args.ssh_key,
        known_hosts=args.known_hosts,
        result_json=args.result_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
