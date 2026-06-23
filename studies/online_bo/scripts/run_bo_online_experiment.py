#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Iterator, TextIO

import numpy as np
import pandas as pd
import yaml


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise RuntimeError(f"Cannot locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization import (  # noqa: E402
    BCLFOptimizationSpec,
    BCLFDataset,
    ContinuousCandidateSpace,
    DiscreteCandidateSpace,
    ExperimentRecorder,
    OptimizerBoTorch,
    ReplaySimulator,
    ServerQueueSimulator,
)
from simulation.result_store import BandpassFilterNamingStrategy, FilesystemResultStore  # noqa: E402


class TeeStream:
    """Write output to both the terminal and a log file."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


@contextlib.contextmanager
def tee_stdout_stderr(log_file: Path | None) -> Iterator[None]:
    if log_file is None:
        yield
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", buffering=1) as fh:
        stdout = TeeStream(sys.stdout, fh)
        stderr = TeeStream(sys.stderr, fh)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def resolve_path(path: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


@dataclass(frozen=True)
class BOOnlineExperimentConfig:
    repo_root: Path = REPO_ROOT
    backend: str = "server"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_init: int = 30
    budget: int = 50
    pool_size: int = 1500
    resolution_mm: float = 0.01
    retrain_every: int = 1
    method_label: str = "global_gp_qlogei"
    run_id_prefix: str = "bo_online_general"
    config_path: Path = REPO_ROOT / "configs" / "filter_bnn_online_uaw_uaw.yaml"
    replay_dataset_path: Path = REPO_ROOT / "data" / "filter_bnn_online_uaw_uaw_dataset.pkl"
    output_dir: Path = REPO_ROOT / "experiments" / "bo_online_general"
    cache_dir: Path = REPO_ROOT / "data" / "bandpass_coupled_line_filter_export" / "ext"
    queue_root: Path = REPO_ROOT / "queue"
    server_job_timeout_sec: float = 0.0
    server_wait_on_failed: bool = True
    template_name: str = "bclf_template_v1"
    design_name: str | None = None
    setup_name: str | None = None
    sweep_name: str | None = None
    resume: bool = True
    reset_checkpoint: bool = False
    clear_queue_artifacts_on_reset: bool = True
    debug_verbose: bool = True
    debug_print_every: int = 1
    dry_run: bool = False
    log_file: Path | None = None

    @property
    def target_trials(self) -> int:
        return int(self.n_init + self.budget)

    @property
    def unit_map(self) -> dict[str, str]:
        return {
            "L": "mm",
            "L1": "mm",
            "L2": "mm",
            "W": "mm",
            "W1": "mm",
            "W2": "mm",
            "S1": "mm",
            "S2": "mm",
        }


class BOOnlineExperimentRunner:
    def __init__(self, config: BOOnlineExperimentConfig) -> None:
        self.config = config
        self.cfg = self._load_yaml_config(config.config_path)
        self.variable_order = list(BCLFDataset.DEFAULT_FILTER_VARIABLE_ORDER)
        self.variable_bounds = self._load_variable_bounds()
        self.spec = BCLFOptimizationSpec(
            passband_ghz=(4.8, 5.4),
            lower_stopband_ghz=(3.6, 4.4),
            upper_stopband_ghz=(5.8, 8.7),
            min_passband_s21_db=-5.0,
            max_passband_s11_db=-7.0,
            max_stopband_s21_db=-15.0,
            s11_weight=0.5,
            stopband_weight=0.5,
        )

    def run(self) -> int:
        self._configure_process_io()
        if not self.config.dry_run:
            self._ensure_output_dirs()
        self._log_header()
        if self.config.dry_run:
            self._log_planned_jobs()
            self._log("dry_run: true; no optimizer will be started.")
            return 0

        campaign_summary_df = self.load_campaign_summary()
        seed_history_frames: list[pd.DataFrame] = []
        latest_history_df = pd.DataFrame()

        self._log(f"target_trials: {self.config.target_trials}")
        self._log(f"pool_size: {self.config.pool_size}")
        self._log(f"resolution_mm: {self.config.resolution_mm}")
        self._log(f"campaign seeds: {self.config.seeds}")

        for seed in self.config.seeds:
            seed = int(seed)
            checkpoint_path = self.checkpoint_for_seed(seed)
            checkpoint_history = self._checkpoint_history(seed)
            checkpoint_df = self._history_frame(seed, checkpoint_history)

            if self.is_seed_completed(seed):
                self._log(
                    f"[Campaign] seed={seed} completed; skipping "
                    f"({len(checkpoint_df)}/{self.config.target_trials} trials)."
                )
                campaign_summary_df = self.upsert_campaign_row(
                    self._summary_row_for_seed(seed, "completed", checkpoint_df)
                )
                seed_history_frames.append(checkpoint_df)
                latest_history_df = checkpoint_df
                continue

            status_label = "resuming" if checkpoint_path.exists() and len(checkpoint_df) else "starting"
            optimizer_seed = self.effective_seed_for_rerun(seed)
            self._log(
                f"[Campaign] seed={seed} optimizer_seed={optimizer_seed} {status_label}; "
                f"checkpoint={checkpoint_path}"
            )
            optimizer = self.build_optimizer(seed)
            self._log(f"checkpoint_path: {optimizer.checkpoint_path} exists={optimizer.checkpoint_path.exists()}")

            run_result = optimizer.run()
            latest_history_df = pd.DataFrame(run_result.history).sort_values("iteration").reset_index(drop=True)
            latest_history_df["seed"] = seed
            seed_history_frames.append(latest_history_df)

            final_status = "completed" if len(latest_history_df) >= self.config.target_trials else "in_progress"
            campaign_summary_df = self.upsert_campaign_row(
                self._summary_row_for_seed(seed, final_status, latest_history_df)
            )

            self._log(f"run_id: {run_result.run_id}")
            self._log(f"best_objective: {run_result.state.best_objective}")
            self._log(f"json: {run_result.recorder_output.json_path}")
            self._log(f"csv: {run_result.recorder_output.csv_path}")

        all_history_df = (
            pd.concat([df for df in seed_history_frames if not df.empty], ignore_index=True)
            if any(not df.empty for df in seed_history_frames)
            else pd.DataFrame()
        )
        self._log("BO logEI campaign summary:")
        self._log(campaign_summary_df.to_string(index=False) if not campaign_summary_df.empty else "<empty>")
        if not latest_history_df.empty:
            self._log("Latest seed history tail:")
            self._log(latest_history_df.tail(10).to_string(index=False))
        self._log(f"combined_history_rows: {len(all_history_df)}")
        return 0

    def campaign_summary_path(self) -> Path:
        return self.config.output_dir / "campaign_summary.csv"

    def run_id_for_seed(self, seed: int) -> str:
        return f"{self.config.run_id_prefix}_seed{int(seed)}"

    def checkpoint_for_seed(self, seed: int) -> Path:
        return self.config.output_dir / f"{self.run_id_for_seed(seed)}.checkpoint.json"

    def effective_seed_for_rerun(self, seed: int) -> int:
        payload = self._load_checkpoint_payload(seed)
        if payload is None or self.config.reset_checkpoint:
            return int(seed)
        checkpoint_seed = payload.get("seed")
        return int(seed) if checkpoint_seed is None else int(checkpoint_seed)

    def is_seed_completed(self, seed: int) -> bool:
        if self.config.reset_checkpoint:
            return False
        return len(self._checkpoint_history(seed)) >= self.config.target_trials

    def load_campaign_summary(self, path: Path | None = None) -> pd.DataFrame:
        path = self.campaign_summary_path() if path is None else Path(path)
        if path.exists():
            df = pd.read_csv(path)
            if "seed" in df.columns:
                df = df.sort_values("seed").reset_index(drop=True)
            return df
        return pd.DataFrame(
            columns=[
                "seed",
                "run_id",
                "method",
                "status",
                "num_trials",
                "best_objective",
                "first_feasible_iteration",
                "cache_hit_ratio",
                "runtime_engine_sec",
                "runtime_sim_sec",
                "json",
                "csv",
                "checkpoint",
                "updated_at",
            ]
        )

    def upsert_campaign_row(self, row: dict[str, Any], path: Path | None = None) -> pd.DataFrame:
        path = self.campaign_summary_path() if path is None else Path(path)
        df = self.load_campaign_summary(path)
        seed = int(row["seed"])
        if "seed" in df.columns and (df["seed"] == seed).any():
            for key, value in row.items():
                df.loc[df["seed"] == seed, key] = value
        elif df.empty:
            df = pd.DataFrame([row])
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.sort_values("seed").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

    def build_optimizer(self, seed: int) -> OptimizerBoTorch:
        run_id = self.run_id_for_seed(seed)
        checkpoint_path = self.checkpoint_for_seed(seed)
        optimizer_seed = self.effective_seed_for_rerun(seed)

        if self.config.reset_checkpoint:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if self.config.clear_queue_artifacts_on_reset:
                removed = self._cleanup_queue_artifacts_for_run(self.config.queue_root, run_id)
                self._log(f"[ResetCleanup] run_id={run_id} {removed}")

        if self.config.backend == "server":
            candidate_space = ContinuousCandidateSpace(
                variable_order=self.variable_order,
                variable_bounds=self.variable_bounds,
                pool_size=self.config.pool_size,
                seed=optimizer_seed,
                resolution=self.config.resolution_mm,
            )
            result_store = FilesystemResultStore(
                root_dir=self.config.cache_dir,
                naming_strategy=BandpassFilterNamingStrategy(),
                port_num=2,
            )
            simulator = ServerQueueSimulator(
                queue_root=str(self.config.queue_root),
                variable_order=self.variable_order,
                unit_map=self.config.unit_map,
                template_name=self.config.template_name,
                port_num=2,
                result_store=result_store,
                design_name=self.config.design_name,
                setup_name=self.config.setup_name,
                sweep_name=self.config.sweep_name,
                metrics_to_extract=["max_s11_db", "min_s21_db"],
                run_id=run_id,
                mm_resolution=self.config.resolution_mm,
                job_timeout_sec=self.config.server_job_timeout_sec,
                wait_on_failed=self.config.server_wait_on_failed,
                verbose=self.config.debug_verbose,
            )
        elif self.config.backend == "replay":
            freq_cfg = self.cfg["data"]["freqs"]
            freqs = np.arange(freq_cfg["min"], freq_cfg["max"], freq_cfg["step"], dtype=float)
            replay_dataset = BCLFDataset.from_pickle(self.config.replay_dataset_path, freqs=freqs)
            candidate_space = DiscreteCandidateSpace(
                variable_order=self.variable_order,
                candidate_geometries=replay_dataset.X_pool,
                match_atol=replay_dataset.match_atol,
                match_rtol=replay_dataset.match_rtol,
            )
            simulator = ReplaySimulator(dataset=replay_dataset)
        else:
            raise ValueError(f"Unsupported backend={self.config.backend!r}. Use server or replay.")

        return OptimizerBoTorch(
            candidate_space=candidate_space,
            simulator=simulator,
            optimization_spec=self.spec,
            recorder=ExperimentRecorder(output_dir=self.config.output_dir),
            budget=self.config.budget,
            n_init=self.config.n_init,
            retrain_every=self.config.retrain_every,
            seed=optimizer_seed,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            resume=(self.config.resume and checkpoint_path.exists()),
            progress_callback=self._progress_debug_callback,
        )

    def _load_yaml_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"config_path does not exist: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_variable_bounds(self) -> dict[str, tuple[float, float]]:
        reg_min = list(self.cfg["data"]["regularization"]["min"])
        reg_max = list(self.cfg["data"]["regularization"]["max"])
        if len(reg_min) == len(self.variable_order) + 1:
            reg_min = reg_min[: len(self.variable_order)]
            reg_max = reg_max[: len(self.variable_order)]
        if len(reg_min) != len(self.variable_order):
            raise ValueError(
                f"Regularization dimension mismatch: len(min)={len(reg_min)} "
                f"variable_order={len(self.variable_order)}"
            )
        return {
            name: (float(reg_min[i]), float(reg_max[i]))
            for i, name in enumerate(self.variable_order)
        }

    def _configure_process_io(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(line_buffering=True)

    def _ensure_output_dirs(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _log_header(self) -> None:
        self._log(f"[BOOnline] started: {datetime.now(timezone.utc).isoformat()}")
        self._log(f"repo_root: {self.config.repo_root}")
        self._log(f"config: {self.config.config_path} exists={self.config.config_path.exists()}")
        self._log(f"output: {self.config.output_dir}")
        self._log(f"campaign_summary: {self.campaign_summary_path()}")
        self._log(f"method: {self.config.method_label}")
        self._log(f"backend: {self.config.backend}")
        self._log(f"seeds: {self.config.seeds}")
        self._log(f"target_trials: {self.config.target_trials}")
        self._log(f"pool_size: {self.config.pool_size}")
        self._log(f"resolution_mm: {self.config.resolution_mm}")
        self._log("variable_bounds:")
        bounds_df = pd.DataFrame(
            {
                "variable": self.variable_order,
                "low": [self.variable_bounds[k][0] for k in self.variable_order],
                "high": [self.variable_bounds[k][1] for k in self.variable_order],
            }
        )
        self._log(bounds_df.to_string(index=False))

    def _log_planned_jobs(self) -> None:
        rows = []
        for seed in self.config.seeds:
            rows.append(
                {
                    "seed": int(seed),
                    "run_id": self.run_id_for_seed(seed),
                    "n_init": int(self.config.n_init),
                    "budget": int(self.config.budget),
                    "target_trials": int(self.config.target_trials),
                    "pool_size": int(self.config.pool_size),
                    "resolution_mm": float(self.config.resolution_mm),
                    "checkpoint": str(self.checkpoint_for_seed(seed)),
                }
            )
        self._log(pd.DataFrame(rows).to_string(index=False))

    def _load_checkpoint_payload(self, seed: int) -> dict[str, Any] | None:
        checkpoint_path = self.checkpoint_for_seed(seed)
        if not checkpoint_path.exists():
            return None
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))

    def _checkpoint_history(self, seed: int) -> list[dict[str, Any]]:
        payload = self._load_checkpoint_payload(seed)
        if payload is None:
            return []
        history = payload.get("history", [])
        return history if isinstance(history, list) else []

    def _history_frame(self, seed: int, history: list[dict[str, Any]]) -> pd.DataFrame:
        hist = pd.DataFrame(history)
        if hist.empty:
            return hist
        hist = hist.copy()
        hist["seed"] = int(seed)
        if "iteration" in hist:
            hist["iteration"] = pd.to_numeric(hist["iteration"], errors="coerce")
            hist = hist[np.isfinite(hist["iteration"])].copy()
            hist["iteration"] = hist["iteration"].astype(int)
            hist = hist.sort_values("iteration").reset_index(drop=True)
        return hist

    def _summary_row_for_seed(self, seed: int, status: str, history_df: pd.DataFrame) -> dict[str, Any]:
        run_id = self.run_id_for_seed(seed)
        return {
            "seed": int(seed),
            "run_id": run_id,
            "method": self.config.method_label,
            "status": status,
            "num_trials": int(len(history_df)),
            "best_objective": self._best_objective_from_history(history_df),
            "first_feasible_iteration": self._first_feasible_iteration(history_df),
            "cache_hit_ratio": self._cache_hit_ratio(history_df),
            "runtime_engine_sec": self._runtime_value(history_df, "t_engine_cum_sec"),
            "runtime_sim_sec": self._runtime_value(history_df, "t_simulator_cum_sec"),
            "json": str(self.config.output_dir / f"{run_id}.json"),
            "csv": str(self.config.output_dir / f"{run_id}.csv"),
            "checkpoint": str(self.checkpoint_for_seed(seed)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _first_feasible_iteration(self, history_df: pd.DataFrame) -> int | None:
        if history_df.empty or "feasible" not in history_df or "iteration" not in history_df:
            return None
        feasible_idx = history_df.index[history_df["feasible"] == True].tolist()  # noqa: E712
        return int(history_df.loc[feasible_idx[0], "iteration"]) if feasible_idx else None

    def _best_objective_from_history(self, history_df: pd.DataFrame) -> float:
        if history_df.empty:
            return float("nan")
        if "best_so_far" in history_df:
            best = pd.to_numeric(history_df["best_so_far"], errors="coerce")
            if np.isfinite(best).any():
                return float(np.nanmin(best))
        if "objective" in history_df:
            objective = pd.to_numeric(history_df["objective"], errors="coerce")
            if np.isfinite(objective).any():
                return float(np.nanmin(objective))
        return float("nan")

    def _cache_hit_ratio(self, history_df: pd.DataFrame) -> float:
        if history_df.empty or "cache_hit" not in history_df:
            return float("nan")
        return float(history_df["cache_hit"].fillna(False).astype(bool).mean())

    def _runtime_value(self, history_df: pd.DataFrame, column: str) -> float:
        if history_df.empty or column not in history_df:
            return float("nan")
        values = pd.to_numeric(history_df[column], errors="coerce")
        finite = values[np.isfinite(values)]
        return float(finite.iloc[-1]) if len(finite) else float("nan")

    def _progress_debug_callback(self, trial: dict[str, Any], history: list[dict[str, Any]]) -> None:
        if not self.config.debug_verbose:
            return
        iteration = int(trial.get("iteration", -1))
        if self.config.debug_print_every > 1 and iteration % int(self.config.debug_print_every) != 0:
            return
        self._log(
            "[Optimizer] "
            f"iter={iteration:04d} "
            f"phase={trial.get('phase')} "
            f"obj={trial.get('objective')} "
            f"best={trial.get('best_so_far')} "
            f"status={trial.get('simulation_status')} "
            f"cache_hit={trial.get('cache_hit')} "
            f"job_id={trial.get('job_id')} "
            f"history={len(history)}"
        )

    def _cleanup_queue_artifacts_for_run(self, queue_root: Path, run_id: str) -> dict[str, int]:
        removed = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for state in ("pending", "running", "done", "failed"):
            state_dir = queue_root / state
            if not state_dir.exists():
                continue
            suffix = ".result.json" if state in ("done", "failed") else ".json"
            for artifact_path in state_dir.glob(f"{run_id}_it*{suffix}"):
                artifact_path.unlink(missing_ok=True)
                removed[state] += 1
        return removed

    def _log(self, message: Any) -> None:
        print(str(message), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BO online BCLF experiment as a normal Python process.",
    )
    parser.add_argument("--backend", choices=["server", "replay"], default="server")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-init", type=int, default=30)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=1500)
    parser.add_argument("--resolution-mm", type=float, default=0.01, help="Geometry snap/submission resolution in mm.")
    parser.add_argument("--retrain-every", type=int, default=1)
    parser.add_argument("--run-id-prefix", default="bo_online_general")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/bo_online_general"))
    parser.add_argument("--config-path", type=Path, default=Path("configs/filter_bnn_online_uaw_uaw.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/bandpass_coupled_line_filter_export/ext"))
    parser.add_argument("--queue-root", type=Path, default=Path("queue"))
    parser.add_argument(
        "--server-job-timeout-sec",
        type=float,
        default=0.0,
        help="Server queue wait timeout in seconds; <=0 waits indefinitely for the current job.",
    )
    parser.add_argument(
        "--server-wait-on-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When server backend sees failed/<job>.result.json, keep waiting instead of advancing to the next candidate.",
    )
    parser.add_argument("--replay-dataset-path", type=Path, default=Path("data/filter_bnn_online_uaw_uaw_dataset.pkl"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument(
        "--no-clear-queue-artifacts",
        action="store_false",
        dest="clear_queue_artifacts_on_reset",
        default=True,
    )
    parser.add_argument("--debug-print-every", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> BOOnlineExperimentConfig:
    return BOOnlineExperimentConfig(
        repo_root=REPO_ROOT,
        backend=args.backend,
        seeds=[int(seed) for seed in args.seeds],
        n_init=int(args.n_init),
        budget=int(args.budget),
        pool_size=int(args.pool_size),
        resolution_mm=float(args.resolution_mm),
        retrain_every=int(args.retrain_every),
        run_id_prefix=str(args.run_id_prefix),
        config_path=resolve_path(args.config_path),
        replay_dataset_path=resolve_path(args.replay_dataset_path),
        output_dir=resolve_path(args.output_dir),
        cache_dir=resolve_path(args.cache_dir),
        queue_root=resolve_path(args.queue_root),
        server_job_timeout_sec=float(args.server_job_timeout_sec),
        server_wait_on_failed=bool(args.server_wait_on_failed),
        resume=bool(args.resume),
        reset_checkpoint=bool(args.reset_checkpoint),
        clear_queue_artifacts_on_reset=bool(args.clear_queue_artifacts_on_reset),
        debug_verbose=not bool(args.quiet),
        debug_print_every=int(args.debug_print_every),
        dry_run=bool(args.dry_run),
        log_file=None if args.log_file is None else resolve_path(args.log_file),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    for name in ("n_init", "budget", "pool_size", "retrain_every", "debug_print_every"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if float(args.resolution_mm) <= 0:
        parser.error("--resolution-mm must be > 0")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    config = config_from_args(args)
    with tee_stdout_stderr(config.log_file):
        try:
            return BOOnlineExperimentRunner(config).run()
        except KeyboardInterrupt:
            print("[BOOnline] interrupted by KeyboardInterrupt", flush=True)
            return 130
        except Exception:
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
