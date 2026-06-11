from __future__ import annotations

"""Generate and optionally run a local Spiral Inductor candidate-space dataset."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.io_utils import read_json, write_json  # noqa: E402
from optimization.candidate_space import DiscreteCandidateSpace  # noqa: E402
from optimization.metrics import (  # noqa: E402
    DifferentialInductanceAtFrequencyMetric,
    DifferentialQAtFrequencyMetric,
    DifferentialSRFMetric,
    SParameterAtFrequencyMetric,
    compute_sparameter_metric_values,
)
from optimization.sparam_response import SParameterResponse  # noqa: E402
from worker.param_strings import build_spiral_inductor_export_filename  # noqa: E402
from worker.schemas import JobRequest  # noqa: E402


INDUCTOR_VARIABLE_ORDER = ["Din", "W", "S", "H"]
INDUCTOR_UNIT_MAP = {"Din": "um", "W": "um", "S": "um", "H": "um"}
DEFAULT_INDUCTOR_RESOLUTION_UM: dict[str, float | None] = {
    "Din": 0.1,
    "W": 0.1,
    "S": 0.1,
    "H": None,
}

DEFAULT_TEMPLATE_NAME = "spiral_inductor_t3_template_v1"
DEFAULT_DESIGN_NAME = "SpiralInductorT3"
DEFAULT_SETUP_NAME = "Setup1"
DEFAULT_SWEEP_NAME = "Sweep"
DEFAULT_EXPORT_DIR = Path("D:/ansys/spiral_inductor_s_parameters_export/ext")
DEFAULT_WORKER_CONFIG = REPO_ROOT / "configs" / "worker.yaml"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / ".runtime"
DEFAULT_QUEUE_ROOT = DEFAULT_RUNTIME_ROOT / "queue"
DEFAULT_METRICS_TO_EXTRACT = ["max_s11_db", "min_s21_db"]

SUMMARY_COLUMNS = [
    "job_id",
    "status",
    "Din",
    "W",
    "S",
    "H",
    "resolution_um",
    "output_filename",
    "touchstone",
    "result_json",
    "error_message",
    "max_s11_db",
    "min_s21_db",
    "ports",
    "freq_start_GHz",
    "freq_stop_GHz",
    "covers_target_GHz",
    "L_5GHz_nH",
    "Q_5GHz",
    "SRF_GHz",
    "S11_5GHz_dB",
    "S21_5GHz_dB",
    "target_L_error_nH",
    "target_rank",
]


@dataclass(frozen=True)
class InductorDesignBounds:
    din_min_um: float = 50.0
    din_max_um: float = 70.0
    w_min_um: float = 12.0
    w_max_um: float = 18.0
    s_min_um: float = 1.0
    s_max_um: float = 2.0
    h_um: float = 2.0

    def validate(self) -> None:
        if self.din_min_um > self.din_max_um:
            raise ValueError("din_min_um must be <= din_max_um")
        if self.w_min_um > self.w_max_um:
            raise ValueError("w_min_um must be <= w_max_um")
        if self.s_min_um > self.s_max_um:
            raise ValueError("s_min_um must be <= s_max_um")
        if self.h_um <= 0:
            raise ValueError("h_um must be > 0")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local Spiral Inductor candidate-space queue, optionally run "
            "the single-thread local worker, and summarize exported Touchstone files."
        )
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=None,
        help="Local offline queue root. Defaults to .runtime/queue.",
    )
    parser.add_argument("--worker-config", type=Path, default=DEFAULT_WORKER_CONFIG)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--run-worker", action="store_true", help="Actually run HFSS through run_local_queue_worker.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned jobs without writing queue files.")
    parser.add_argument("--preserve-local-project-copy", action="store_true")
    parser.add_argument("--target-ghz", type=float, default=5.0)
    parser.add_argument("--target-l-nh", type=float, default=1.2)
    parser.add_argument("--template-name", default=DEFAULT_TEMPLATE_NAME)
    parser.add_argument("--design-name", default=DEFAULT_DESIGN_NAME)
    parser.add_argument("--setup-name", default=DEFAULT_SETUP_NAME)
    parser.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--din-min-um", type=float, default=50.0)
    parser.add_argument("--din-max-um", type=float, default=70.0)
    parser.add_argument("--w-min-um", type=float, default=12.0)
    parser.add_argument("--w-max-um", type=float, default=18.0)
    parser.add_argument("--s-min-um", type=float, default=1.0)
    parser.add_argument("--s-max-um", type=float, default=2.0)
    parser.add_argument("--h-um", type=float, default=2.0)
    parser.add_argument("--din-resolution-um", type=float, default=DEFAULT_INDUCTOR_RESOLUTION_UM["Din"])
    parser.add_argument("--w-resolution-um", type=float, default=DEFAULT_INDUCTOR_RESOLUTION_UM["W"])
    parser.add_argument("--s-resolution-um", type=float, default=DEFAULT_INDUCTOR_RESOLUTION_UM["S"])
    return parser.parse_args(argv)


def build_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    run_id = args.run_id or f"inductor_candidate_space_{_stamp()}"
    if args.runtime_root is not None:
        runtime_root = args.runtime_root
    elif args.queue_root is not None:
        runtime_root = args.queue_root.parent
    else:
        runtime_root = DEFAULT_RUNTIME_ROOT
    queue_root = args.queue_root or (runtime_root / "queue")
    results_root = queue_root / "results"
    return {
        "run_id": run_id,
        "runtime_root": runtime_root,
        "queue_root": queue_root,
        "logs_dir": runtime_root / "logs" / run_id,
        "template_copy_root": runtime_root / "aedt_copies" / run_id,
        "worker_config": runtime_root / "worker_configs" / f"{run_id}.yaml",
        "jobs_json": results_root / f"{run_id}_inductor_candidate_jobs.json",
        "summary_csv": results_root / f"{run_id}_inductor_candidate_summary.csv",
        "summary_json": results_root / f"{run_id}_inductor_candidate_summary.json",
        "top_csv": results_root / f"{run_id}_inductor_top_candidates.csv",
    }


def build_bounds(args: argparse.Namespace) -> InductorDesignBounds:
    return InductorDesignBounds(
        din_min_um=args.din_min_um,
        din_max_um=args.din_max_um,
        w_min_um=args.w_min_um,
        w_max_um=args.w_max_um,
        s_min_um=args.s_min_um,
        s_max_um=args.s_max_um,
        h_um=args.h_um,
    )


def build_resolution(args: argparse.Namespace) -> dict[str, float | None]:
    return normalize_inductor_resolution(
        {
            "Din": args.din_resolution_um,
            "W": args.w_resolution_um,
            "S": args.s_resolution_um,
            "H": None,
        }
    )


def normalize_inductor_resolution(
    resolution_um: dict[str, float | None] | None = None,
) -> dict[str, float | None]:
    resolution = dict(DEFAULT_INDUCTOR_RESOLUTION_UM)
    if resolution_um is not None:
        resolution.update(resolution_um)
    normalized: dict[str, float | None] = {}
    for name in INDUCTOR_VARIABLE_ORDER:
        value = resolution.get(name)
        if value is None:
            normalized[name] = None
            continue
        step = float(value)
        if step <= 0:
            raise ValueError(f"Resolution for {name} must be positive or None")
        normalized[name] = step
    return normalized


def format_um(value: float) -> str:
    return f"{float(value):.2f}um"


def format_resolution_um(resolution_um: dict[str, float | None]) -> str:
    parts: list[str] = []
    for name in INDUCTOR_VARIABLE_ORDER:
        value = resolution_um.get(name)
        parts.append(f"{name}=fixed" if value is None else f"{name}={value:g}um")
    return ";".join(parts)


def generate_candidate_rows(
    *,
    count: int,
    seed: int,
    bounds: InductorDesignBounds,
    resolution_um: dict[str, float | None],
) -> list[dict[str, float]]:
    geometries = generate_lhs_geometries(
        count=count,
        seed=seed,
        bounds=bounds,
        variable_order=list(INDUCTOR_VARIABLE_ORDER),
        resolution_um=resolution_um,
    )
    candidate_space = DiscreteCandidateSpace(
        variable_order=list(INDUCTOR_VARIABLE_ORDER),
        candidate_geometries=geometries,
    )
    rows: list[dict[str, float]] = []
    for geometry in candidate_space.remaining():
        rows.append({name: float(value) for name, value in zip(INDUCTOR_VARIABLE_ORDER, geometry)})
    return rows


def generate_lhs_geometries(
    *,
    count: int,
    seed: int | None,
    bounds: InductorDesignBounds | None = None,
    variable_order: list[str] | None = None,
    resolution_um: dict[str, float | None] | None = None,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    bounds = bounds or InductorDesignBounds()
    bounds.validate()
    order = list(variable_order or INDUCTOR_VARIABLE_ORDER)
    if order != INDUCTOR_VARIABLE_ORDER:
        raise ValueError(f"Inductor LHS expects variable_order={INDUCTOR_VARIABLE_ORDER}, got {order}")

    resolution = normalize_inductor_resolution(resolution_um)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    batch_count = max(count * 2, count + 8)
    for _ in range(1000):
        candidates = _raw_lhs_geometries(count=batch_count, bounds=bounds, rng=rng)
        for row in candidates:
            snapped = snap_inductor_geometry(row, variable_order=order, resolution_um=resolution)
            try:
                validate_inductor_geometry(snapped, variable_order=order)
            except ValueError:
                continue
            if not _is_within_bounds(snapped, bounds=bounds):
                continue
            key = tuple(round(float(value), 10) for value in snapped)
            if key in seen:
                continue
            seen.add(key)
            rows.append(snapped)
            if len(rows) == count:
                return np.asarray(rows, dtype=float)
    raise ValueError(
        f"Unable to generate {count} unique valid inductor geometries with resolution={resolution}"
    )


def validate_inductor_geometry(
    geometry: np.ndarray | list[float],
    *,
    variable_order: list[str] | None = None,
) -> None:
    values = geometry_to_variables(geometry, variable_order=variable_order or INDUCTOR_VARIABLE_ORDER)
    if values["Din"] <= 2.0 * (values["W"] + values["S"]):
        raise ValueError("Spiral geometry invalid: Din must be > 2*(W+S)")
    if values["H"] <= 0:
        raise ValueError("Spiral geometry invalid: H must be > 0")


def geometry_to_variables(
    geometry: np.ndarray | list[float],
    *,
    variable_order: list[str] | None = None,
) -> dict[str, float]:
    order = list(variable_order or INDUCTOR_VARIABLE_ORDER)
    values = np.asarray(geometry, dtype=float).reshape(-1)
    if len(values) != len(order):
        raise ValueError(f"geometry dimension mismatch: got {len(values)}, expected {len(order)}")
    return {name: float(values[index]) for index, name in enumerate(order)}


def snap_inductor_geometry(
    geometry: np.ndarray | list[float],
    *,
    variable_order: list[str] | None = None,
    resolution_um: dict[str, float | None] | None = None,
) -> np.ndarray:
    order = list(variable_order or INDUCTOR_VARIABLE_ORDER)
    values = geometry_to_variables(geometry, variable_order=order)
    resolution = normalize_inductor_resolution(resolution_um)
    snapped: list[float] = []
    for name in order:
        value = float(values[name])
        step = resolution.get(name)
        if step is not None:
            value = round(round(value / step) * step, 10)
        snapped.append(value)
    return np.asarray(snapped, dtype=float)


def _raw_lhs_geometries(
    *,
    count: int,
    bounds: InductorDesignBounds,
    rng: np.random.Generator,
) -> np.ndarray:
    unit = _latin_hypercube(count=count, dim=3, rng=rng)
    rows = np.empty((count, len(INDUCTOR_VARIABLE_ORDER)), dtype=float)
    rows[:, 0] = _scale(unit[:, 0], bounds.din_min_um, bounds.din_max_um)
    rows[:, 1] = _scale(unit[:, 1], bounds.w_min_um, bounds.w_max_um)
    rows[:, 2] = _scale(unit[:, 2], bounds.s_min_um, bounds.s_max_um)
    rows[:, 3] = float(bounds.h_um)
    return rows


def _latin_hypercube(*, count: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    sample = np.empty((count, dim), dtype=float)
    for col in range(dim):
        strata = rng.permutation(count)
        jitter = rng.random(count)
        sample[:, col] = (strata + jitter) / float(count)
    return sample


def _is_within_bounds(row: np.ndarray, *, bounds: InductorDesignBounds) -> bool:
    values = geometry_to_variables(row)
    return (
        bounds.din_min_um <= values["Din"] <= bounds.din_max_um
        and bounds.w_min_um <= values["W"] <= bounds.w_max_um
        and bounds.s_min_um <= values["S"] <= bounds.s_max_um
        and bounds.h_um == values["H"]
    )


def _scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return float(low) + np.asarray(values, dtype=float) * (float(high) - float(low))


def build_output_filename(index: int, row: dict[str, float]) -> str:
    return build_spiral_inductor_export_filename(
        row["Din"],
        row["S"],
        row["W"],
        counter=index,
        counter_width=3,
        value_precision=2,
    )


def build_job_payload(
    *,
    run_id: str,
    index: int,
    row: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    job_id = f"{run_id}_it{index:04d}"
    payload = {
        "job_id": job_id,
        "template_name": args.template_name,
        "design_name": args.design_name,
        "setup_name": args.setup_name,
        "sweep_name": args.sweep_name,
        "variables": {
            "Din": format_um(row["Din"]),
            "W": format_um(row["W"]),
            "S": format_um(row["S"]),
            "H": format_um(row["H"]),
        },
        "output_filename": build_output_filename(index, row),
        "metrics_to_extract": list(DEFAULT_METRICS_TO_EXTRACT),
        "artifact_options": {"export_touchstone": True, "export_csv": False},
        "submit_time": datetime.now(timezone.utc).isoformat(),
        "retry_count": 0,
    }
    JobRequest.model_validate(payload)
    return payload


def build_worker_config_payload(
    *,
    base_worker_config: Path,
    queue_root: Path,
    export_dir: Path,
    logs_dir: Path,
    template_copy_root: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(base_worker_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in worker config: {base_worker_config}")
    payload = dict(payload)
    payload.update(
        {
            "worker_name": f"{payload.get('worker_name', 'hfss_worker')}_inductor_local",
            "queue_root": str(queue_root),
            "results_dir": str(export_dir),
            "logs_dir": str(logs_dir),
            "copy_template_per_job": True,
            "template_copy_root": str(template_copy_root),
            "preserve_local_project_copy": False,
            "new_desktop": True,
            "close_desktop_after_job": True,
            "default_export_touchstone": True,
            "default_export_csv": False,
        }
    )
    return payload


def prepare_run(args: argparse.Namespace) -> tuple[dict[str, Path | str], list[dict[str, Any]]]:
    paths = build_paths(args)
    run_id = str(paths["run_id"])
    bounds = build_bounds(args)
    resolution = build_resolution(args)
    resolution_label = format_resolution_um(resolution)
    candidate_rows = generate_candidate_rows(
        count=args.count,
        seed=args.seed,
        bounds=bounds,
        resolution_um=resolution,
    )
    jobs = [
        {
            **row,
            "job_id": f"{run_id}_it{index:04d}",
            "resolution_um": resolution_label,
            "output_filename": build_output_filename(index, row),
            "payload": build_job_payload(run_id=run_id, index=index, row=row, args=args),
        }
        for index, row in enumerate(candidate_rows)
    ]
    if len({job["output_filename"] for job in jobs}) != len(jobs):
        raise ValueError("Generated output filenames are not unique")
    return paths, jobs


def write_queue_and_config(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path | str],
    jobs: list[dict[str, Any]],
) -> None:
    queue_root = Path(paths["queue_root"])
    export_dir = args.export_dir
    resolution = build_resolution(args)
    logs_dir = Path(paths["logs_dir"])
    template_copy_root = Path(paths["template_copy_root"])
    for path in (
        queue_root / "pending",
        queue_root / "running",
        queue_root / "done",
        queue_root / "failed",
        queue_root / "results",
        export_dir,
        logs_dir,
        template_copy_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    worker_payload = build_worker_config_payload(
        base_worker_config=args.worker_config,
        queue_root=queue_root,
        export_dir=export_dir,
        logs_dir=logs_dir,
        template_copy_root=template_copy_root,
    )
    worker_config_path = Path(paths["worker_config"])
    worker_config_path.parent.mkdir(parents=True, exist_ok=True)
    worker_config_path.write_text(
        yaml.safe_dump(worker_payload, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    for job in jobs:
        write_json(queue_root / "pending" / f"{job['job_id']}.json", job["payload"])

    write_json(
        paths["jobs_json"],
        {
            "run_id": paths["run_id"],
            "queue_root": str(queue_root),
            "export_dir": str(export_dir),
            "worker_config": str(paths["worker_config"]),
            "count": len(jobs),
            "seed": args.seed,
            "variable_order": list(INDUCTOR_VARIABLE_ORDER),
            "resolution_um": resolution,
            "geometry_rule": "Din > 2*(W+S)",
            "jobs": [_public_job_row(job) for job in jobs],
        },
    )


def run_local_worker(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path | str],
    max_jobs: int,
) -> int:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "worker" / "run_local_queue_worker.py"),
        "--worker-config",
        str(paths["worker_config"]),
        "--queue-root",
        str(paths["queue_root"]),
        "--logs-dir",
        str(paths["logs_dir"]),
        "--template-copy-root",
        str(paths["template_copy_root"]),
        "--max-jobs",
        str(max_jobs),
        "--no-wait",
    ]
    if args.preserve_local_project_copy:
        command.append("--preserve-local-project-copy")

    print("worker_command=", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    return int(completed.returncode)


def collect_results(
    *,
    paths: dict[str, Path | str],
    jobs: list[dict[str, Any]],
    target_ghz: float,
    target_l_nh: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    queue_root = Path(paths["queue_root"])
    for job in jobs:
        state, result_path, payload = _result_payload(queue_root, str(job["job_id"]))
        artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        touchstone_path = _resolve_artifact_path(
            queue_root,
            artifacts.get("touchstone") if isinstance(artifacts, dict) else None,
        )
        row = {
            **_public_job_row(job),
            "status": state,
            "result_json": str(result_path) if result_path else None,
            "touchstone": str(touchstone_path) if touchstone_path else None,
            "error_message": payload.get("error_message") if isinstance(payload, dict) else None,
            "max_s11_db": metrics.get("max_s11_db") if isinstance(metrics, dict) else None,
            "min_s21_db": metrics.get("min_s21_db") if isinstance(metrics, dict) else None,
            "ports": None,
            "freq_start_GHz": None,
            "freq_stop_GHz": None,
            "covers_target_GHz": None,
            "L_5GHz_nH": None,
            "Q_5GHz": None,
            "SRF_GHz": None,
            "S11_5GHz_dB": None,
            "S21_5GHz_dB": None,
            "target_L_error_nH": None,
            "target_rank": None,
        }
        if state == "done" and touchstone_path is not None and touchstone_path.exists():
            row.update(_extract_touchstone_summary(touchstone_path, target_ghz=target_ghz))
            if row["L_5GHz_nH"] is not None:
                row["target_L_error_nH"] = abs(float(row["L_5GHz_nH"]) - float(target_l_nh))
        rows.append(row)
    _assign_target_rank(rows)
    return rows


def write_summary_outputs(
    *,
    paths: dict[str, Path | str],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    worker_returncode: int | None,
) -> None:
    summary_csv = Path(paths["summary_csv"])
    summary_json = Path(paths["summary_json"])
    top_csv = Path(paths["top_csv"])
    resolution = build_resolution(args)
    write_csv(summary_csv, rows, SUMMARY_COLUMNS)
    top_rows = _top_candidate_rows(rows, limit=10)
    write_csv(top_csv, top_rows, SUMMARY_COLUMNS)
    write_json(
        summary_json,
        {
            "run_id": paths["run_id"],
            "queue_root": str(paths["queue_root"]),
            "export_dir": str(args.export_dir),
            "worker_config": str(paths["worker_config"]),
            "resolution_um": resolution,SkyuySk
            "worker_returncode": worker_returncode,
            "status_counts": _status_counts(rows),
            "summary_csv": str(summary_csv),
            "top_csv": str(top_csv),
            "top_candidates": top_rows,
            "rows": rows,
        },
    )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(paths: dict[str, Path | str], rows: list[dict[str, Any]]) -> None:
    print(f"run_id={paths['run_id']}")
    print(f"queue_root={paths['queue_root']}")
    print(f"summary_csv={paths['summary_csv']}")
    print(f"summary_json={paths['summary_json']}")
    print(f"status_counts={_status_counts(rows)}")
    top_rows = _top_candidate_rows(rows, limit=5)
    if top_rows:
        print("\ntop_candidates:")
        print_table(
            top_rows,
            ["target_rank", "Din", "W", "S", "L_5GHz_nH", "Q_5GHz", "SRF_GHz", "touchstone"],
        )


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    widths = {
        col: max(len(col), *(len(_format_cell(row.get(col))) for row in rows))
        for col in columns
    }
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(_format_cell(row.get(col)).ljust(widths[col]) for col in columns))


def run(args: argparse.Namespace) -> int:
    paths, jobs = prepare_run(args)
    if args.dry_run:
        print_table([_public_job_row(job) for job in jobs], ["job_id", "Din", "W", "S", "H", "output_filename"])
        return 0

    write_queue_and_config(args=args, paths=paths, jobs=jobs)
    worker_returncode = None
    if args.run_worker:
        worker_returncode = run_local_worker(args=args, paths=paths, max_jobs=len(jobs))

    rows = collect_results(
        paths=paths,
        jobs=jobs,
        target_ghz=args.target_ghz,
        target_l_nh=args.target_l_nh,
    )
    write_summary_outputs(paths=paths, rows=rows, args=args, worker_returncode=worker_returncode)
    print_summary(paths, rows)
    return worker_returncode if worker_returncode is not None else 0


def _public_job_row(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "Din": float(job["Din"]),
        "W": float(job["W"]),
        "S": float(job["S"]),
        "H": float(job["H"]),
        "resolution_um": job.get("resolution_um"),
        "output_filename": job["output_filename"],
    }


def _result_payload(queue_root: Path, job_id: str) -> tuple[str, Path | None, dict[str, Any] | None]:
    for state in ("done", "failed"):
        path = queue_root / state / f"{job_id}.result.json"
        if path.exists():
            return state, path, read_json(path)
    return "pending_or_missing", None, None


def _resolve_artifact_path(queue_root: Path, artifact: str | None) -> Path | None:
    if not artifact:
        return None
    path = Path(artifact)
    return path if path.is_absolute() else queue_root / path


def _extract_touchstone_summary(path: Path, *, target_ghz: float) -> dict[str, Any]:
    try:
        import skrf as rf

        network = rf.Network(str(path))
        freq_start = float(network.f[0] / 1e9)
        freq_stop = float(network.f[-1] / 1e9)
        target_hz = float(target_ghz) * 1e9
        response = SParameterResponse.from_sparams(network.f, network.s)
        extracted = compute_sparameter_metric_values(
            response,
            [
                DifferentialInductanceAtFrequencyMetric(target_hz),
                DifferentialQAtFrequencyMetric(target_hz),
                DifferentialSRFMetric(),
                SParameterAtFrequencyMetric("S11", target_hz),
                SParameterAtFrequencyMetric("S21", target_hz),
            ],
        )
        return {
            "ports": int(network.number_of_ports),
            "freq_start_GHz": freq_start,
            "freq_stop_GHz": freq_stop,
            "covers_target_GHz": bool(network.f[0] <= target_ghz * 1e9 <= network.f[-1]),
            **extracted,
        }
    except Exception as exc:
        return {
            "error_message": f"touchstone_parse_failure: {exc}",
        }


def _assign_target_rank(rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        [
            row
            for row in rows
            if row.get("status") == "done"
            and row.get("target_L_error_nH") is not None
            and row.get("Q_5GHz") is not None
        ],
        key=lambda row: (float(row["target_L_error_nH"]), -float(row["Q_5GHz"])),
    )
    for rank, row in enumerate(ranked, start=1):
        row["target_rank"] = rank


def _top_candidate_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ranked = [row for row in rows if row.get("target_rank") is not None]
    return sorted(ranked, key=lambda row: int(row["target_rank"]))[:limit]


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
