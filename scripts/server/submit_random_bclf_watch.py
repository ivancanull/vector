from __future__ import annotations

# Recommended coherent server-side test command:
#
#   cd /home/zhanghf/projects/vector
#   PYTHONPATH=src:. python scripts/server/submit_random_bclf_watch.py \
#     --queue-root queue \
#     --count 5 \
#     --seed 123 \
#     --timeout-sec 7200 \
#     --poll-interval-sec 10
#
# Defaults align to your original HFSS export names: Design=HFSSDesign1,
# Setup=Setup1, Solution/Sweep=Sweep.

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.io_utils import read_json  # noqa: E402
from common.paths import ensure_runtime_dirs  # noqa: E402
from server.experiment_adapter import to_batch_simulation_job  # noqa: E402
from server.experiment_models import (  # noqa: E402
    CaseVariant,
    DesignSpace,
    Experiment,
    ExperimentObjective,
    VariableSpec,
)
from server.queue_manager import QueueManager  # noqa: E402


@dataclass(frozen=True)
class BclfVariableRanges:
    l_min: float = 7.7
    l_max: float = 7.9
    l1_min: float = 8.0
    l1_max: float = 8.2
    l2_min: float = 7.8
    l2_max: float = 8.0
    w_min: float = 0.9
    w_max: float = 1.1
    w1_min: float = 0.7
    w1_max: float = 0.9
    w2_min: float = 0.9
    w2_max: float = 1.1
    s1_min: float = 0.13
    s1_max: float = 0.16
    s2_min: float = 0.55
    s2_max: float = 0.6
    unit: str = "mm"

    def validate(self) -> None:
        _validate_range("L", self.l_min, self.l_max)
        _validate_range("L1", self.l1_min, self.l1_max)
        _validate_range("L2", self.l2_min, self.l2_max)
        _validate_range("W", self.w_min, self.w_max)
        _validate_range("W1", self.w1_min, self.w1_max)
        _validate_range("W2", self.w2_min, self.w2_max)
        _validate_range("S1", self.s1_min, self.s1_max)
        _validate_range("S2", self.s2_min, self.s2_max)
        if not self.unit.strip():
            raise ValueError("unit must be non-empty")


@dataclass(frozen=True)
class ResultSummary:
    job_id: str
    status: str
    variables: dict[str, str]
    metrics: dict[str, Any] | None
    error_message: str | None
    artifacts: dict[str, Any]
    result_path: Path


def _validate_range(name: str, low: float, high: float) -> None:
    if low > high:
        raise ValueError(f"{name} min must be <= max")


def _str_p(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = text + ".0"
    return text.replace("-", "m").replace(".", "d")


def _now_batch_id() -> str:
    return "bclf_random_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _default_design_space(ranges: BclfVariableRanges) -> DesignSpace:
    ranges.validate()
    return DesignSpace(
        variable_specs={
            "L": VariableSpec(name="L", unit=ranges.unit, constraint_type="range", min_value=ranges.l_min, max_value=ranges.l_max),
            "L1": VariableSpec(name="L1", unit=ranges.unit, constraint_type="range", min_value=ranges.l1_min, max_value=ranges.l1_max),
            "L2": VariableSpec(name="L2", unit=ranges.unit, constraint_type="range", min_value=ranges.l2_min, max_value=ranges.l2_max),
            "W": VariableSpec(name="W", unit=ranges.unit, constraint_type="range", min_value=ranges.w_min, max_value=ranges.w_max),
            "W1": VariableSpec(name="W1", unit=ranges.unit, constraint_type="range", min_value=ranges.w1_min, max_value=ranges.w1_max),
            "W2": VariableSpec(name="W2", unit=ranges.unit, constraint_type="range", min_value=ranges.w2_min, max_value=ranges.w2_max),
            "S1": VariableSpec(name="S1", unit=ranges.unit, constraint_type="range", min_value=ranges.s1_min, max_value=ranges.s1_max),
            "S2": VariableSpec(name="S2", unit=ranges.unit, constraint_type="range", min_value=ranges.s2_min, max_value=ranges.s2_max),
        }
    )


def generate_random_bclf_experiment(
    *,
    batch_id: str,
    count: int,
    seed: int | None,
    ranges: BclfVariableRanges,
    template_name: str,
    design_name: str | None,
    setup_name: str | None,
    sweep_name: str | None,
) -> Experiment:
    if count <= 0:
        raise ValueError("count must be positive")

    ds = _default_design_space(ranges)
    rng = random.Random(seed)

    cases: list[CaseVariant] = []
    for index in range(1, count + 1):
        l = rng.uniform(ranges.l_min, ranges.l_max)
        l1 = rng.uniform(ranges.l1_min, ranges.l1_max)
        l2 = rng.uniform(ranges.l2_min, ranges.l2_max)
        w = rng.uniform(ranges.w_min, ranges.w_max)
        w1 = rng.uniform(ranges.w1_min, ranges.w1_max)
        w2 = rng.uniform(ranges.w2_min, ranges.w2_max)
        s1 = rng.uniform(ranges.s1_min, ranges.s1_max)
        s2 = rng.uniform(ranges.s2_min, ranges.s2_max)

        case_id = f"c{index:02d}"
        output_filename = (
            "Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter"
            f"_L_{_str_p(l)}_L1_{_str_p(l1)}_L2_{_str_p(l2)}"
            f"_W_{_str_p(w)}_W1_{_str_p(w1)}_W2_{_str_p(w2)}"
            f"_S1_{_str_p(s1)}_S2_{_str_p(s2)}.s2p"
        )
        cases.append(
            CaseVariant(
                case_id=case_id,
                variables={
                    "L": f"{l:.2f}{ranges.unit}",
                    "L1": f"{l1:.2f}{ranges.unit}",
                    "L2": f"{l2:.2f}{ranges.unit}",
                    "W": f"{w:.2f}{ranges.unit}",
                    "W1": f"{w1:.2f}{ranges.unit}",
                    "W2": f"{w2:.2f}{ranges.unit}",
                    "S1": f"{s1:.2f}{ranges.unit}",
                    "S2": f"{s2:.2f}{ranges.unit}",
                },
                output_filename=output_filename,
            )
        )

    experiment = Experiment(
        experiment_id=batch_id,
        template_name=template_name,
        design_name=design_name,
        setup_name=setup_name,
        sweep_name=sweep_name,
        design_space=ds,
        objectives=[
            ExperimentObjective(metric_name="max_s11_db", direction="minimize"),
            ExperimentObjective(metric_name="min_s21_db", direction="maximize"),
        ],
        cases=cases,
        artifact_options={"export_touchstone": True, "export_csv": False},
    )
    experiment.validate_cases_against_design_space()
    return experiment


def generate_random_bclf_batch(
    *,
    batch_id: str,
    count: int,
    seed: int | None,
    ranges: BclfVariableRanges,
    template_name: str,
    design_name: str | None,
    setup_name: str | None,
    sweep_name: str | None,
):
    experiment = generate_random_bclf_experiment(
        batch_id=batch_id,
        count=count,
        seed=seed,
        ranges=ranges,
        template_name=template_name,
        design_name=design_name,
        setup_name=setup_name,
        sweep_name=sweep_name,
    )
    return to_batch_simulation_job(experiment)


def submit_random_batch(queue_root: Path, batch) -> list[Path]:
    ensure_runtime_dirs(queue_root, REPO_ROOT / "logs")
    manager = QueueManager(queue_root)
    return manager.enqueue_batch(batch)


def _job_id_for(batch_id: str, case: Any) -> str:
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


def print_summary(
    *,
    expected_jobs: dict[str, dict[str, str]],
    finished: dict[str, ResultSummary],
    missing: set[str],
) -> None:
    print("")
    print("Final random BCLF batch summary")
    print("job_id,status,L,L1,L2,W,W1,W2,S1,S2,result")
    for job_id in sorted(expected_jobs):
        variables = expected_jobs[job_id]
        summary = finished.get(job_id)
        if summary is None:
            status = "missing"
            result_text = "missing_result"
        else:
            status = summary.status
            if summary.status == "done":
                result_text = ", ".join(f"{k}={v}" for k, v in sorted((summary.metrics or {}).items()))
            else:
                result_text = f"error={summary.error_message}"

        print(
            f"{job_id},{status},{variables.get('L')},{variables.get('L1')},{variables.get('L2')},"
            f"{variables.get('W')},{variables.get('W1')},{variables.get('W2')},"
            f"{variables.get('S1')},{variables.get('S2')},{result_text}"
        )

    if missing:
        print("")
        print("Timed out waiting for:")
        for job_id in sorted(missing):
            print(job_id)


def export_done_touchstones(
    *,
    queue_root: Path,
    finished: dict[str, ResultSummary],
    export_dir: Path,
) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for summary in finished.values():
        if summary.status != "done":
            continue
        touchstone_rel = summary.artifacts.get("touchstone")
        if not isinstance(touchstone_rel, str) or not touchstone_rel:
            continue
        src = queue_root / Path(touchstone_rel)
        if not src.exists():
            continue
        dst = export_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit random BCLF jobs and watch for worker results."
    )
    parser.add_argument("--queue-root", type=Path, default=REPO_ROOT / "queue")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "data" / "bandpass_coupled_line_filter_export" / "ext",
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--poll-interval-sec", type=int, default=10)

    parser.add_argument("--template-name", default="bclf_template_v1")
    parser.add_argument("--design-name", default="HFSSDesign1")
    parser.add_argument("--setup-name", default="Setup1")
    parser.add_argument("--sweep-name", default="Sweep")

    parser.add_argument("--unit", default="mm")
    parser.add_argument("--l-min", type=float, default=7.7)
    parser.add_argument("--l-max", type=float, default=7.9)
    parser.add_argument("--l1-min", type=float, default=8.0)
    parser.add_argument("--l1-max", type=float, default=8.2)
    parser.add_argument("--l2-min", type=float, default=7.8)
    parser.add_argument("--l2-max", type=float, default=8.0)
    parser.add_argument("--w-min", type=float, default=0.9)
    parser.add_argument("--w-max", type=float, default=1.1)
    parser.add_argument("--w1-min", type=float, default=0.7)
    parser.add_argument("--w1-max", type=float, default=0.9)
    parser.add_argument("--w2-min", type=float, default=0.9)
    parser.add_argument("--w2-max", type=float, default=1.1)
    parser.add_argument("--s1-min", type=float, default=0.13)
    parser.add_argument("--s1-max", type=float, default=0.16)
    parser.add_argument("--s2-min", type=float, default=0.55)
    parser.add_argument("--s2-max", type=float, default=0.6)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    batch_id = args.batch_id or _now_batch_id()
    ranges = BclfVariableRanges(
        l_min=args.l_min,
        l_max=args.l_max,
        l1_min=args.l1_min,
        l1_max=args.l1_max,
        l2_min=args.l2_min,
        l2_max=args.l2_max,
        w_min=args.w_min,
        w_max=args.w_max,
        w1_min=args.w1_min,
        w1_max=args.w1_max,
        w2_min=args.w2_min,
        w2_max=args.w2_max,
        s1_min=args.s1_min,
        s1_max=args.s1_max,
        s2_min=args.s2_min,
        s2_max=args.s2_max,
        unit=args.unit,
    )

    try:
        batch = generate_random_bclf_batch(
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
    print(
        "setup_sweep_override="
        f"{batch.setup_name if batch.setup_name else '<template-default>'}/"
        f"{batch.sweep_name if batch.sweep_name else '<template-default>'}"
    )
    for path in emitted:
        print(f"submitted={path}")
    print(f"watching={len(expected_jobs)} timeout_sec={args.timeout_sec}")

    finished, missing = watch_results(
        queue_root=args.queue_root,
        expected_jobs=expected_jobs,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    copied = export_done_touchstones(
        queue_root=args.queue_root,
        finished=finished,
        export_dir=args.export_dir,
    )
    for path in copied:
        print(f"exported={path}")
    print_summary(expected_jobs=expected_jobs, finished=finished, missing=missing)
    return 1 if missing else 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
