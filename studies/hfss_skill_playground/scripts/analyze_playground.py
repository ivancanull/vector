from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from {current}")


REPO_ROOT = find_repo_root()
STUDY_ROOT = REPO_ROOT / "studies" / "hfss_skill_playground"
DATA_ROOT = REPO_ROOT / "experiments" / "hfss_skill_playground"
BOOLEAN_REPRO_FIELDS = (
    "syntax_pass",
    "import_pass",
    "hfss_build_success",
    "setup_success",
    "solve_success",
    "result_export_success",
    "verifier_pass",
)
SUMMARY_FIELDS = (
    "case_id",
    "method",
    "seed",
    "syntax_pass",
    "import_pass",
    "hfss_build_success",
    "setup_success",
    "solve_success",
    "result_export_success",
    "verifier_pass",
    "human_edit_count",
    "repair_count",
    "failure_type",
    "hfss_call_count",
    "final_best_true_objective",
    "target_reached",
    "calls_to_target",
    "wall_clock_time_to_target_sec",
    "wall_clock_total_sec",
    "area_under_best_so_far",
    "failure_adjusted_score",
)


@dataclass(frozen=True)
class CaseObjective:
    direction: str
    target_threshold: float | None


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_case_objectives(root: Path) -> dict[str, CaseObjective]:
    data = load_yaml(root / "cases.yaml")
    objectives: dict[str, CaseObjective] = {}
    for case in data.get("selected_cases", []):
        objective = case.get("objective", {})
        threshold = objective.get("target_threshold")
        objectives[str(case["id"])] = CaseObjective(
            direction=str(objective.get("direction", "minimize")),
            target_threshold=None if threshold is None else float(threshold),
        )
    return objectives


def load_failure_penalties(root: Path) -> dict[str, float]:
    data = load_yaml(root / "failure_taxonomy.yaml")
    return {str(row["id"]): float(row["penalty"]) for row in data.get("failure_types", [])}


def evaluation_paths(root: Path) -> list[Path]:
    return sorted((root / "runs").glob("*/*/*/evaluation.json"))


def read_history_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def numeric(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "done", "pass", "passed"}


def normalize_trials(evaluation_path: Path, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    optimization = evaluation.get("optimization", {})
    trials = optimization.get("trials")
    if isinstance(trials, list):
        return [dict(row) for row in trials]
    history_file = optimization.get("history_file") or evaluation.get("history_file")
    if history_file:
        path = Path(history_file)
        if not path.is_absolute():
            path = evaluation_path.parent / path
        return read_history_from_csv(path)
    return read_history_from_csv(evaluation_path.parent / "history.csv")


def best_so_far_values(trials: list[dict[str, Any]], direction: str) -> list[float]:
    values: list[float] = []
    running: float | None = None
    for row in trials:
        candidate = numeric(row.get("best_so_far"))
        if not math.isfinite(candidate):
            candidate = numeric(row.get("true_objective"), numeric(row.get("objective")))
        if not math.isfinite(candidate):
            continue
        if running is None:
            running = candidate
        elif direction == "maximize":
            running = max(running, candidate)
        else:
            running = min(running, candidate)
        values.append(float(running))
    return values


def target_reached(value: float, objective: CaseObjective) -> bool:
    if objective.target_threshold is None or not math.isfinite(value):
        return False
    if objective.direction == "maximize":
        return value >= objective.target_threshold
    return value <= objective.target_threshold


def first_target_hit(
    trials: list[dict[str, Any]],
    values: list[float],
    objective: CaseObjective,
) -> tuple[int | None, float | None]:
    for idx, value in enumerate(values):
        if target_reached(value, objective):
            row = trials[idx] if idx < len(trials) else {}
            call = int(numeric(row.get("call"), idx + 1))
            wall = numeric(row.get("wall_clock_sec"), numeric(row.get("t_wall_cum_sec")))
            return call, None if not math.isfinite(wall) else float(wall)
    return None, None


def area_under_curve(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_run(
    evaluation_path: Path,
    objectives: dict[str, CaseObjective],
    penalties: dict[str, float],
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    case_id = str(evaluation.get("case_id") or evaluation_path.parts[-4])
    method = str(evaluation.get("method") or evaluation_path.parts[-3])
    seed = str(evaluation.get("seed") or evaluation_path.parts[-2])
    reproduction = dict(evaluation.get("reproduction", {}))
    failure_type = reproduction.get("failure_type") or evaluation.get("failure_type")
    objective = objectives.get(case_id, CaseObjective(direction="minimize", target_threshold=None))
    trials = normalize_trials(evaluation_path, evaluation)
    values = best_so_far_values(trials, objective.direction)
    final_best = values[-1] if values else math.nan
    calls_to_target, wall_to_target = first_target_hit(trials, values, objective)
    wall_total = numeric(evaluation.get("wall_clock_total_sec"))
    if not math.isfinite(wall_total) and trials:
        wall_total = max(
            (
                numeric(row.get("wall_clock_sec"), numeric(row.get("t_wall_cum_sec")))
                for row in trials
            ),
            default=math.nan,
        )
    penalty = penalties.get(str(failure_type), 0.0) if failure_type else 0.0
    if not boolish(reproduction.get("solve_success", False)):
        penalty = max(penalty, penalties.get("solve_failure", 500.0))
    if not boolish(reproduction.get("result_export_success", False)):
        penalty = max(penalty, penalties.get("export_failure", 400.0))
    if not values:
        penalty = max(penalty, penalties.get("metric_extraction_failure", 300.0))
    reached = target_reached(final_best, objective)
    if not reached:
        penalty = max(penalty, penalties.get("target_not_reached", 100.0))

    row = {
        "case_id": case_id,
        "method": method,
        "seed": seed,
        "human_edit_count": int(numeric(reproduction.get("human_edit_count"), 0)),
        "repair_count": int(numeric(reproduction.get("repair_count"), 0)),
        "failure_type": "" if failure_type is None else str(failure_type),
        "hfss_call_count": len(values),
        "final_best_true_objective": final_best,
        "target_reached": reached,
        "calls_to_target": "" if calls_to_target is None else calls_to_target,
        "wall_clock_time_to_target_sec": "" if wall_to_target is None else wall_to_target,
        "wall_clock_total_sec": "" if not math.isfinite(wall_total) else wall_total,
        "area_under_best_so_far": "" if area_under_curve(values) is None else area_under_curve(values),
        "failure_adjusted_score": (final_best if math.isfinite(final_best) else 0.0) + penalty,
    }
    for field in BOOLEAN_REPRO_FIELDS:
        row[field] = boolish(reproduction.get(field, False))
    return row


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 if boolish(row.get(field)) else 0.0 for row in rows) / len(rows)


def finite_median(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [numeric(row.get(field)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else float(median(finite))


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures = Counter(row.get("failure_type") or "none" for row in rows)
    for row in rows:
        by_method[str(row["method"])].append(row)

    lines = [
        "# HFSS RF Skill Playground Report",
        "",
        f"Runs analyzed: {len(rows)}",
        "",
        "## Method Summary",
        "",
        "| Method | Runs | Median final objective | Median calls to target | Solve success | Export success | Verifier pass | Median failure-adjusted score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, method_rows in sorted(by_method.items()):
        lines.append(
            "| {method} | {runs} | {final} | {calls} | {solve:.2f} | {export:.2f} | {verifier:.2f} | {score} |".format(
                method=method,
                runs=len(method_rows),
                final=_fmt(finite_median(method_rows, "final_best_true_objective")),
                calls=_fmt(finite_median(method_rows, "calls_to_target")),
                solve=rate(method_rows, "solve_success"),
                export=rate(method_rows, "result_export_success"),
                verifier=rate(method_rows, "verifier_pass"),
                score=_fmt(finite_median(method_rows, "failure_adjusted_score")),
            )
        )

    lines.extend(["", "## Failure Counts", ""])
    for failure, count in failures.most_common():
        lines.append(f"- `{failure}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HFSS RF skill playground runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Deprecated compatibility alias: use one root for study files and run data.",
    )
    parser.add_argument("--study-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.root is not None:
        study_root = args.root.resolve()
        data_root = args.root.resolve()
    else:
        study_root = (args.study_root or STUDY_ROOT).resolve()
        data_root = (args.data_root or DATA_ROOT).resolve()

    objectives = load_case_objectives(study_root)
    penalties = load_failure_penalties(study_root)
    rows = [summarize_run(path, objectives, penalties) for path in evaluation_paths(data_root)]
    summary_path = args.summary or (data_root / "summary.csv")
    report_path = args.report or (study_root / "reports" / "playground_report.generated.md")
    write_summary(rows, summary_path)
    write_report(rows, report_path)
    print(f"Analyzed {len(rows)} runs")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
