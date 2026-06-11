from __future__ import annotations

"""Launch multiple isolated fetch/run/submit worker slots.

Each slot runs the existing fetch_run_submit_job.py entrypoint with its own
local queue, logs, results directory, and per-job AEDT copy root.
"""

import argparse
from dataclasses import dataclass
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

from common.io_utils import read_yaml  # noqa: E402
from scripts.worker import fetch_run_submit_job  # noqa: E402


DEFAULT_SLOT_ROOT = REPO_ROOT / ".runtime" / "parallel_worker_slots"


@dataclass(frozen=True)
class ParallelWorkerSlot:
    index: int
    name: str
    root: Path
    worker_config: Path


def build_slot_config_payload(
    base_config: dict[str, Any],
    *,
    slot_index: int,
    slot_root: Path,
) -> dict[str, Any]:
    slot_name = f"slot_{slot_index:02d}"
    root = slot_root / slot_name
    queue_root = root / "queue"
    payload = dict(base_config)
    base_worker_name = str(base_config.get("worker_name", "hfss_worker"))
    payload.update(
        {
            "worker_name": f"{base_worker_name}_{slot_name}",
            "queue_root": str(queue_root),
            "results_dir": str(queue_root / "results"),
            "logs_dir": str(root / "logs"),
            "new_desktop": True,
            "close_desktop_after_job": True,
            "copy_template_per_job": True,
            "template_copy_root": str(root / "aedt_copies"),
        }
    )
    return payload


def write_slot_worker_configs(
    *,
    base_worker_config: Path,
    slot_root: Path,
    slots: int,
) -> list[ParallelWorkerSlot]:
    base_config = read_yaml(base_worker_config)
    emitted: list[ParallelWorkerSlot] = []
    for index in range(1, slots + 1):
        name = f"slot_{index:02d}"
        root = slot_root / name
        root.mkdir(parents=True, exist_ok=True)
        payload = build_slot_config_payload(
            base_config,
            slot_index=index,
            slot_root=slot_root,
        )
        config_path = root / "worker.generated.yaml"
        with config_path.open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        emitted.append(
            ParallelWorkerSlot(
                index=index,
                name=name,
                root=root,
                worker_config=config_path,
            )
        )
    return emitted


def build_slot_command(slot: ParallelWorkerSlot, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("fetch_run_submit_job.py")),
        "--worker-config",
        str(slot.worker_config),
        "--server-host",
        args.server_host,
        "--server-port",
        str(args.server_port),
        "--username",
        args.username,
        "--ssh-key",
        str(args.ssh_key),
        "--known-hosts",
        str(args.known_hosts),
        "--remote-queue-root",
        args.remote_queue_root,
        "--sftp-connect-timeout-sec",
        str(args.sftp_connect_timeout_sec),
        "--sftp-connect-attempts",
        str(args.sftp_connect_attempts),
        "--sftp-connect-retry-delay-sec",
        str(args.sftp_connect_retry_delay_sec),
        "--idle-status-interval-sec",
        str(args.idle_status_interval_sec),
    ]
    if args.poll_interval_sec is not None:
        command.extend(["--poll-interval-sec", str(args.poll_interval_sec)])
    if not args.wait_when_empty:
        command.append("--no-wait-when-empty")
    if args.once:
        command.append("--once")
    return command


def run(args: argparse.Namespace) -> int:
    if args.slots <= 0:
        print("invalid_args=slots must be positive", file=sys.stderr)
        return 2

    slot_root = args.slot_root.resolve()
    slot_root.mkdir(parents=True, exist_ok=True)
    slots = write_slot_worker_configs(
        base_worker_config=args.worker_config,
        slot_root=slot_root,
        slots=args.slots,
    )

    processes: list[subprocess.Popen] = []
    try:
        for slot in slots:
            command = build_slot_command(slot, args)
            print(f"slot={slot.name} worker_config={slot.worker_config}")
            print(f"launch={' '.join(command)}")
            processes.append(subprocess.Popen(command, cwd=REPO_ROOT))

        return_codes = [process.wait() for process in processes]
        return 0 if all(code == 0 for code in return_codes) else 1
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple isolated fetch/run/submit worker slots."
    )
    parser.add_argument("--worker-config", type=Path, default=fetch_run_submit_job.DEFAULT_WORKER_CONFIG)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--slot-root", type=Path, default=DEFAULT_SLOT_ROOT)
    parser.add_argument("--server-host", default=fetch_run_submit_job.DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=fetch_run_submit_job.DEFAULT_SERVER_PORT)
    parser.add_argument("--username", default=fetch_run_submit_job.DEFAULT_USERNAME)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--remote-queue-root", default=fetch_run_submit_job.DEFAULT_REMOTE_QUEUE_ROOT)
    parser.add_argument(
        "--sftp-connect-timeout-sec",
        type=int,
        default=fetch_run_submit_job.DEFAULT_SFTP_CONNECT_TIMEOUT_SEC,
        help="Per-attempt SSH/SFTP connection timeout forwarded to each slot.",
    )
    parser.add_argument(
        "--sftp-connect-attempts",
        type=int,
        default=fetch_run_submit_job.DEFAULT_SFTP_CONNECT_ATTEMPTS,
        help="SSH/SFTP connection retry count forwarded to each slot.",
    )
    parser.add_argument(
        "--sftp-connect-retry-delay-sec",
        type=float,
        default=fetch_run_submit_job.DEFAULT_SFTP_CONNECT_RETRY_DELAY_SEC,
        help="Delay between SSH/SFTP connection retries forwarded to each slot.",
    )
    parser.add_argument(
        "--wait-when-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to each worker slot.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=None,
        help="Forwarded to each worker slot.",
    )
    parser.add_argument(
        "--idle-status-interval-sec",
        type=int,
        default=60,
        help="Forwarded to each worker slot.",
    )
    parser.add_argument("--once", action="store_true", help="Each slot processes one job and exits.")
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
