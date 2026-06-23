from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise RuntimeError(f"Cannot locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_EXPECTED_BUDGETED_TRIALS = 80
DEFAULT_OWAFS_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BNN_SEEDS = (0, 1, 2)
DEFAULT_COMPARISON_RELATIVE_DIR = Path("experiments/online_lhsdb_warmup/owafs_bnn/comparison")
FORBIDDEN_BNN_PATH_TOKENS = ("online_lhs15_shared", "lhs15_shared", "previous_lhs", "previous-lhs")


@dataclass(frozen=True)
class MethodConfig:
    method: str
    relative_dir: Path
    file_pattern: str
    required_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RunMetrics:
    method: str
    seed: int
    source_csv: Path
    raw_rows: int
    budgeted_rows: int
    validation_rows: int
    final_best_objective: float
    best_iteration: int
    best_phase: str
    best_case_id: str
    best_after_15: float
    best_after_30: float
    best_after_50: float
    best_after_80: float
    improvement_15_to_80: float
    cache_hit_ratio: float
    runtime_engine_sec: float
    runtime_sim_sec: float
    best_so_far: np.ndarray


DEFAULT_METHOD_CONFIGS = (
    MethodConfig(
        method="OWAFS",
        relative_dir=Path("experiments/online_lhsdb_warmup/owafs_bnn"),
        file_pattern="owafs_bnn_online_lhsdb_warmup_seed{seed}.csv",
        required_seeds=DEFAULT_OWAFS_SEEDS,
    ),
    MethodConfig(
        method="BO",
        relative_dir=Path("experiments/bo_online_general"),
        file_pattern="bo_online_general_seed*.csv",
    ),
    MethodConfig(
        method="LHS",
        relative_dir=Path("experiments/lhs_online_general"),
        file_pattern="lhs_online_general_seed*.csv",
    ),
    MethodConfig(
        method="BNN",
        relative_dir=Path("experiments/bnn_online_general"),
        file_pattern="bnn_online_general_seed{seed}.csv",
        required_seeds=DEFAULT_BNN_SEEDS,
    ),
)


def parse_bool_series(values: pd.Series, *, default_for_missing: bool) -> pd.Series:
    def parse_value(value: object) -> bool:
        if pd.isna(value):
            return default_for_missing
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")

    return values.map(parse_value).astype(bool)


def extract_seed(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if match is None:
        raise RuntimeError(f"Could not infer seed from filename: {path}")
    return int(match.group(1))


def validate_bnn_path(path: Path) -> None:
    lowered = str(path).lower()
    for token in FORBIDDEN_BNN_PATH_TOKENS:
        if token in lowered:
            raise RuntimeError(
                f"Forbidden BNN baseline path contains {token!r}: {path}. "
                "Use experiments/bnn_online_general only."
            )


class OWAFSOnlineGeneralComparer:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        output_dir: Path | None = None,
        expected_budgeted_trials: int = DEFAULT_EXPECTED_BUDGETED_TRIALS,
        method_configs: Sequence[MethodConfig] = DEFAULT_METHOD_CONFIGS,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.output_dir = Path(output_dir) if output_dir is not None else self.repo_root / DEFAULT_COMPARISON_RELATIVE_DIR
        self.expected_budgeted_trials = int(expected_budgeted_trials)
        self.method_configs = tuple(method_configs)
        self.excluded_incomplete: list[dict[str, object]] = []

    def run(self) -> list[Path]:
        runs = self.load_complete_runs()
        per_run = self._per_run_frame(runs)
        summary = self._summary_frame(per_run)
        ranked = self._ranked_frame(per_run)
        curves = self._curve_summary_frame(runs)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            self.output_dir / "owafs_vs_bo_lhs_bnn_per_run.csv",
            self.output_dir / "owafs_vs_bo_lhs_bnn_summary.csv",
            self.output_dir / "owafs_vs_bo_lhs_bnn_ranked_runs.csv",
            self.output_dir / "owafs_vs_bo_lhs_bnn_best_so_far_by_iteration.csv",
        ]
        per_run.to_csv(outputs[0], index=False)
        summary.to_csv(outputs[1], index=False)
        ranked.to_csv(outputs[2], index=False)
        curves.to_csv(outputs[3], index=False)

        if self.excluded_incomplete:
            excluded_path = self.output_dir / "owafs_vs_bo_lhs_bnn_excluded_incomplete_runs.csv"
            pd.DataFrame(self.excluded_incomplete).to_csv(excluded_path, index=False)
            outputs.append(excluded_path)
        return outputs

    def load_complete_runs(self) -> list[RunMetrics]:
        runs: list[RunMetrics] = []
        for config in self.method_configs:
            config_runs = self._load_config_runs(config)
            if not config_runs:
                raise RuntimeError(f"No complete runs loaded for method {config.method}")
            runs.extend(config_runs)
        self._validate_included_bnn_sources(runs)
        return runs

    def _load_config_runs(self, config: MethodConfig) -> list[RunMetrics]:
        base_dir = self.repo_root / config.relative_dir
        if config.required_seeds is None:
            paths = sorted(base_dir.glob(config.file_pattern))
        else:
            paths = [base_dir / config.file_pattern.format(seed=seed) for seed in config.required_seeds]

        runs = []
        for path in paths:
            if config.method == "BNN":
                validate_bnn_path(path)
            if not path.exists():
                raise RuntimeError(f"Missing {config.method} CSV: {path}")
            rows = pd.read_csv(path)
            budgeted = self._budgeted_rows(rows).reset_index(drop=True)
            if len(budgeted) != self.expected_budgeted_trials:
                record = {
                    "method": config.method,
                    "seed": extract_seed(path),
                    "source_csv": str(path),
                    "budgeted_rows": len(budgeted),
                    "expected_budgeted_trials": self.expected_budgeted_trials,
                }
                if config.required_seeds is not None:
                    raise RuntimeError(
                        f"{config.method} seed {record['seed']} is incomplete: "
                        f"{len(budgeted)}/{self.expected_budgeted_trials} budgeted rows"
                    )
                self.excluded_incomplete.append(record)
                continue
            metrics = self._load_run(path, config.method, rows=rows, budgeted=budgeted)
            runs.append(metrics)
        return runs

    def _load_run(
        self,
        path: Path,
        method: str,
        *,
        rows: pd.DataFrame | None = None,
        budgeted: pd.DataFrame | None = None,
    ) -> RunMetrics:
        rows = pd.read_csv(path) if rows is None else rows
        budgeted = self._budgeted_rows(rows).reset_index(drop=True) if budgeted is None else budgeted
        seed = extract_seed(path)
        objective = pd.to_numeric(budgeted["objective"], errors="coerce").to_numpy(dtype=float)
        objective[~np.isfinite(objective)] = np.inf
        if objective.size == 0 or not np.isfinite(objective).any():
            raise RuntimeError(f"{method} seed {seed} contains no finite budgeted objective values")

        best_so_far = np.minimum.accumulate(objective)
        best_pos = int(np.argmin(objective))
        best_row = budgeted.iloc[best_pos]
        return RunMetrics(
            method=method,
            seed=seed,
            source_csv=path,
            raw_rows=len(rows),
            budgeted_rows=len(budgeted),
            validation_rows=self._validation_count(rows),
            final_best_objective=float(best_so_far[-1]),
            best_iteration=self._row_iteration(best_row, best_pos),
            best_phase=str(best_row.get("phase", "")),
            best_case_id=str(best_row.get("case_id", "")),
            best_after_15=self._best_at(best_so_far, 15),
            best_after_30=self._best_at(best_so_far, 30),
            best_after_50=self._best_at(best_so_far, 50),
            best_after_80=self._best_at(best_so_far, 80),
            improvement_15_to_80=float(self._best_at(best_so_far, 15) - self._best_at(best_so_far, 80)),
            cache_hit_ratio=self._cache_hit_ratio(budgeted),
            runtime_engine_sec=self._runtime_sum(budgeted, "t_engine_sec"),
            runtime_sim_sec=self._runtime_sum(budgeted, "t_simulator_sec"),
            best_so_far=best_so_far,
        )

    def _budgeted_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        if "counts_toward_budget" in rows.columns:
            mask = parse_bool_series(rows["counts_toward_budget"], default_for_missing=True)
        else:
            phase = rows.get("phase", pd.Series("", index=rows.index)).fillna("").astype(str).str.lower()
            mask = phase != "validation"
        return rows[mask].copy()

    def _validation_count(self, rows: pd.DataFrame) -> int:
        if "counts_toward_budget" in rows.columns:
            mask = parse_bool_series(rows["counts_toward_budget"], default_for_missing=True)
            return int((~mask).sum())
        phase = rows.get("phase", pd.Series("", index=rows.index)).fillna("").astype(str).str.lower()
        return int((phase == "validation").sum())

    def _row_iteration(self, row: pd.Series, fallback: int) -> int:
        if "iteration" not in row.index or pd.isna(row["iteration"]):
            return fallback
        return int(row["iteration"])

    def _best_at(self, best_so_far: np.ndarray, trial_count: int) -> float:
        if best_so_far.shape[0] < trial_count:
            raise RuntimeError(
                f"Cannot compute best after {trial_count} trials from {best_so_far.shape[0]} budgeted rows"
            )
        return float(best_so_far[trial_count - 1])

    def _cache_hit_ratio(self, rows: pd.DataFrame) -> float:
        if "cache_hit" not in rows.columns:
            return float("nan")
        return float(parse_bool_series(rows["cache_hit"], default_for_missing=False).mean())

    def _runtime_sum(self, rows: pd.DataFrame, column: str) -> float:
        if column not in rows.columns:
            return float("nan")
        return float(pd.to_numeric(rows[column], errors="coerce").sum())

    def _validate_included_bnn_sources(self, runs: Sequence[RunMetrics]) -> None:
        for run in runs:
            if run.method == "BNN":
                validate_bnn_path(run.source_csv)
                expected_dir = self.repo_root / "experiments" / "bnn_online_general"
                try:
                    run.source_csv.resolve().relative_to(expected_dir.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"BNN baseline must come from {expected_dir}: {run.source_csv}") from exc

    def _per_run_frame(self, runs: Sequence[RunMetrics]) -> pd.DataFrame:
        records = []
        for run in runs:
            records.append(
                {
                    "method": run.method,
                    "seed": run.seed,
                    "source_csv": str(run.source_csv),
                    "raw_rows": run.raw_rows,
                    "budgeted_rows": run.budgeted_rows,
                    "validation_rows": run.validation_rows,
                    "final_best_objective": run.final_best_objective,
                    "best_iteration": run.best_iteration,
                    "best_phase": run.best_phase,
                    "best_case_id": run.best_case_id,
                    "best_after_15": run.best_after_15,
                    "best_after_30": run.best_after_30,
                    "best_after_50": run.best_after_50,
                    "best_after_80": run.best_after_80,
                    "improvement_15_to_80": run.improvement_15_to_80,
                    "cache_hit_ratio": run.cache_hit_ratio,
                    "runtime_engine_sec": run.runtime_engine_sec,
                    "runtime_sim_sec": run.runtime_sim_sec,
                }
            )
        return pd.DataFrame(records).sort_values(["method", "seed"]).reset_index(drop=True)

    def _summary_frame(self, per_run: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for method, group in per_run.groupby("method", sort=False):
            rows.append(
                {
                    "method": method,
                    "n_runs": int(len(group)),
                    "seeds": ",".join(str(seed) for seed in sorted(group["seed"].astype(int).tolist())),
                    "mean_final_best_objective": float(group["final_best_objective"].mean()),
                    "std_final_best_objective": float(group["final_best_objective"].std(ddof=1))
                    if len(group) > 1
                    else float("nan"),
                    "median_final_best_objective": float(group["final_best_objective"].median()),
                    "best_min_objective": float(group["final_best_objective"].min()),
                    "worst_max_objective": float(group["final_best_objective"].max()),
                    "median_best_iteration": float(group["best_iteration"].median()),
                    "median_best_after_15": float(group["best_after_15"].median()),
                    "median_best_after_30": float(group["best_after_30"].median()),
                    "median_best_after_50": float(group["best_after_50"].median()),
                    "median_improvement_15_to_80": float(group["improvement_15_to_80"].median()),
                    "mean_improvement_15_to_80": float(group["improvement_15_to_80"].mean()),
                    "median_cache_hit_ratio": float(group["cache_hit_ratio"].median()),
                    "median_runtime_engine_sec": float(group["runtime_engine_sec"].median()),
                    "median_runtime_sim_sec": float(group["runtime_sim_sec"].median()),
                    "n_final_below_minus20": int((group["final_best_objective"] <= -20.0).sum()),
                    "n_final_below_minus15": int((group["final_best_objective"] <= -15.0).sum()),
                }
            )
        return pd.DataFrame(rows).sort_values("median_final_best_objective").reset_index(drop=True)

    def _ranked_frame(self, per_run: pd.DataFrame) -> pd.DataFrame:
        ranked = per_run.sort_values("final_best_objective").reset_index(drop=True).copy()
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
        return ranked

    def _curve_summary_frame(self, runs: Sequence[RunMetrics]) -> pd.DataFrame:
        rows = []
        for run in runs:
            for iteration, best in enumerate(run.best_so_far):
                rows.append(
                    {
                        "method": run.method,
                        "seed": run.seed,
                        "iteration": iteration,
                        "trial": iteration + 1,
                        "best_so_far": float(best),
                    }
                )
        curve = pd.DataFrame(rows)
        summary_rows = []
        for (method, iteration), group in curve.groupby(["method", "iteration"], sort=False):
            values = group["best_so_far"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "method": method,
                    "iteration": int(iteration),
                    "trial": int(iteration) + 1,
                    "n": int(len(values)),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
        return pd.DataFrame(summary_rows).sort_values(["method", "iteration"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare current OWAFS-BNN against BO/LHS/BNN online_general baselines."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-budgeted-trials", type=int, default=DEFAULT_EXPECTED_BUDGETED_TRIALS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparer = OWAFSOnlineGeneralComparer(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        expected_budgeted_trials=args.expected_budgeted_trials,
    )
    try:
        outputs = comparer.run()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
