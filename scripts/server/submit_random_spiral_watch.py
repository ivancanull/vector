from __future__ import annotations

# Recommended coherent server-side test command:
#
#   cd /home/zhanghf/projects/vector
#   PYTHONPATH=src:. python scripts/server/submit_random_spiral_watch.py \
#     --queue-root queue \
#     --count 5 \
#     --seed 123 \
#     --timeout-sec 7200 \
#     --poll-interval-sec 10
#
# Run the Windows worker SFTP loop at the same time so these server-side
# queue/pending jobs are claimed, simulated, and uploaded back as results.

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import random
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.io_utils import read_json  # noqa: E402
from common.job_types import BatchCase, BatchSimulationJob  # noqa: E402
from common.paths import ensure_runtime_dirs  # noqa: E402
from server.queue_manager import QueueManager  # noqa: E402


DEFAULT_METRICS = ["max_s11_db", "min_s21_db"]


@dataclass(frozen=True)
class RandomSpiralRanges:
    din_min_um: float = 50.0
    din_max_um: float = 70.0
    s_min_um: float = 1.0
    s_max_um: float = 2.0
    w_min_um: float = 12.0
    w_max_um: float = 18.0
    h_um: float = 2.0


@dataclass(frozen=True)
class ResultSummary:
    job_id: str
    status: str
    variables: dict[str, str]
    metrics: dict[str, Any] | None
    error_message: str | None
    artifacts: dict[str, Any]
    result_path: Path


def _now_batch_id() -> str:
    return "spiral_random_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _format_um(value: float) -> str:
    return f"{value:.2f}um"


def _filename_number(value: float) -> str:
    return f"{value:.2f}".replace(".", "d")


def _validate_ranges(ranges: RandomSpiralRanges, count: int) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    if ranges.din_min_um > ranges.din_max_um:
        raise ValueError("din-min-um must be <= din-max-um")
    if ranges.s_min_um > ranges.s_max_um:
        raise ValueError("s-min-um must be <= s-max-um")
    if ranges.w_min_um > ranges.w_max_um:
        raise ValueError("w-min-um must be <= w-max-um")
    if ranges.h_um <= 0:
        raise ValueError("h-um must be positive")


def generate_random_spiral_batch(
    *,
    batch_id: str,
    count: int,
    seed: int | None,
    ranges: RandomSpiralRanges,
    template_name: str,
    design_name: str,
    setup_name: str,
    sweep_name: str,
) -> BatchSimulationJob:
    _validate_ranges(ranges, count)
    rng = random.Random(seed)
    cases: list[BatchCase] = []

    for index in range(1, count + 1):
        din = rng.uniform(ranges.din_min_um, ranges.din_max_um)
        spacing = rng.uniform(ranges.s_min_um, ranges.s_max_um)
        width = rng.uniform(ranges.w_min_um, ranges.w_max_um)
        case_id = f"c{index:02d}"
        output_filename = (
            f"{batch_id}_{case_id}_"
            f"Din{_filename_number(din)}_"
            f"S{_filename_number(spacing)}_"
            f"W{_filename_number(width)}.s2p"
        )
        cases.append(
            BatchCase(
                case_id=case_id,
                variables={
                    "Din": _format_um(din),
                    "H": _format_um(ranges.h_um),
                    "S": _format_um(spacing),
                    "W": _format_um(width),
                },
                output_filename=output_filename,
            )
        )

    return BatchSimulationJob(
        batch_id=batch_id,
        template_name=template_name,
        cases=cases,
        metrics_to_extract=list(DEFAULT_METRICS),
        design_name=design_name,
        setup_name=setup_name,
        sweep_name=sweep_name,
        artifact_options={"export_touchstone": True, "export_csv": True},
    )


def submit_random_batch(queue_root: Path, batch: BatchSimulationJob) -> list[Path]:
    ensure_runtime_dirs(queue_root, REPO_ROOT / "logs")
    manager = QueueManager(queue_root)
    return manager.enqueue_batch(batch)


def _job_id_for(batch_id: str, case: BatchCase) -> str:
    return f"{batch_id}_{case.case_id}"


def _load_result_if_present(queue_root: Path, job_id: str) -> tuple[Path, dict[str, Any]] | None:
    for state in ("done", "failed"):
        path = queue_root / state / f"{job_id}.result.json"
        if path.exists():
            return path, read_json(path)
    return None


def collect_finished_results(
    queue_root: Path,
    expected_jobs: dict[str, dict[str, str]],
) -> dict[str, ResultSummary]:
    finished: dict[str, ResultSummary] = {}
    for job_id, variables in expected_jobs.items():
        loaded = _load_result_if_present(queue_root, job_id)
        if loaded is None:
            continue

        result_path, payload = loaded
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        metrics = payload.get("metrics")
        if metrics is not None and not isinstance(metrics, dict):
            metrics = {"raw": metrics}

        finished[job_id] = ResultSummary(
            job_id=job_id,
            status=str(payload.get("status", result_path.parent.name)),
            variables=variables,
            metrics=metrics,
            error_message=payload.get("error_message"),
            artifacts=artifacts,
            result_path=result_path,
        )
    return finished


def watch_results(
    *,
    queue_root: Path,
    expected_jobs: dict[str, dict[str, str]],
    timeout_sec: int,
    poll_interval_sec: int,
) -> tuple[dict[str, ResultSummary], set[str]]:
    deadline = time.monotonic() + timeout_sec
    seen: set[str] = set()

    while True:
        finished = collect_finished_results(queue_root, expected_jobs)
        for job_id in sorted(set(finished) - seen):
            summary = finished[job_id]
            seen.add(job_id)
            if summary.status == "done":
                print(f"result={job_id} status=done metrics={summary.metrics}")
            else:
                print(
                    f"result={job_id} status={summary.status} "
                    f"error={summary.error_message}"
                )

        if len(finished) == len(expected_jobs):
            return finished, set()

        remaining = set(expected_jobs) - set(finished)
        if time.monotonic() >= deadline:
            return finished, remaining

        time.sleep(poll_interval_sec)


def _metric_text(summary: ResultSummary) -> str:
    if summary.status != "done":
        return f"error={summary.error_message}"
    if not summary.metrics:
        return "metrics={}"
    return ", ".join(f"{key}={value}" for key, value in sorted(summary.metrics.items()))


def print_summary(
    *,
    expected_jobs: dict[str, dict[str, str]],
    finished: dict[str, ResultSummary],
    missing: set[str],
) -> None:
    print("")
    print("Final random spiral batch summary")
    print("job_id,status,Din,H,S,W,result")
    for job_id in sorted(expected_jobs):
        variables = expected_jobs[job_id]
        summary = finished.get(job_id)
        if summary is None:
            result_text = "missing_result"
            status = "missing"
        else:
            result_text = _metric_text(summary)
            status = summary.status
        print(
            f"{job_id},{status},{variables.get('Din')},{variables.get('H')},"
            f"{variables.get('S')},{variables.get('W')},{result_text}"
        )

    if missing:
        print("")
        print("Timed out waiting for:")
        for job_id in sorted(missing):
            print(job_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit five random spiral jobs and watch for worker results."
    )
    parser.add_argument("--queue-root", type=Path, default=REPO_ROOT / "queue")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--poll-interval-sec", type=int, default=10)
    parser.add_argument("--template-name", default="si_template_v1")
    parser.add_argument("--design-name", default="HFSSDesign1")
    parser.add_argument("--setup-name", default="Setup1")
    parser.add_argument("--sweep-name", default="Sweep")
    parser.add_argument("--din-min-um", type=float, default=50.0)
    parser.add_argument("--din-max-um", type=float, default=70.0)
    parser.add_argument("--s-min-um", type=float, default=1.0)
    parser.add_argument("--s-max-um", type=float, default=2.0)
    parser.add_argument("--w-min-um", type=float, default=12.0)
    parser.add_argument("--w-max-um", type=float, default=18.0)
    parser.add_argument("--h-um", type=float, default=2.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    batch_id = args.batch_id or _now_batch_id()
    ranges = RandomSpiralRanges(
        din_min_um=args.din_min_um,
        din_max_um=args.din_max_um,
        s_min_um=args.s_min_um,
        s_max_um=args.s_max_um,
        w_min_um=args.w_min_um,
        w_max_um=args.w_max_um,
        h_um=args.h_um,
    )

    try:
        batch = generate_random_spiral_batch(
            batch_id=batch_id,
            count=args.count,
            seed=args.seed,
            ranges=ranges,
            template_name=args.template_name,
            design_name=args.design_name,
            setup_name=args.setup_name,
            sweep_name=args.sweep_name,
        )
    except ValueError as exc:
        print(f"invalid_args={exc}", file=sys.stderr)
        return 2

    emitted = submit_random_batch(args.queue_root, batch)
    expected_jobs = {
        _job_id_for(batch.batch_id, case): case.variables
        for case in batch.cases
    }

    print(f"batch_id={batch.batch_id}")
    for path in emitted:
        print(f"submitted={path}")
    print(f"watching={len(expected_jobs)} timeout_sec={args.timeout_sec}")

    finished, missing = watch_results(
        queue_root=args.queue_root,
        expected_jobs=expected_jobs,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    print_summary(expected_jobs=expected_jobs, finished=finished, missing=missing)
    return 1 if missing else 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
