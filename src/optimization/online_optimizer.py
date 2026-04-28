from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .acquisition import AcquisitionFunction
from .objectives import Objective
from .recorder import ExperimentRecorder, RecorderOutput
from .simulators import SimulationResult, Simulator
from .surrogates import SurrogateModel


@dataclass
class OnlineOptimizationState:
    evaluated_geometries: list[list[float]]
    candidate_geometries: list[list[float]]
    best_geometry: list[float] | None = None
    best_objective: float | None = None
    step: int = 0


@dataclass(frozen=True)
class OnlineRunResult:
    run_id: str
    best_trial: dict[str, Any]
    history: list[dict[str, Any]]
    recorder_output: RecorderOutput
    state: OnlineOptimizationState
    checkpoint_path: Path


class OnlineBOOptimizer:
    """Online BO controller for real simulators with checkpoint/resume."""

    def __init__(
        self,
        *,
        variable_order: list[str],
        variable_bounds: dict[str, tuple[float, float]],
        pool_size: int,
        simulator: Simulator,
        objective: Objective,
        surrogate: SurrogateModel,
        acquisition: AcquisitionFunction,
        recorder: ExperimentRecorder,
        budget: int,
        retrain_every: int = 1,
        n_init: int = 0,
        seed: int | None = None,
        run_id: str = "bo_online",
        checkpoint_path: str | Path = "experiments/bo_online/checkpoint.json",
        resume: bool = False,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        if n_init < 0:
            raise ValueError("n_init must be >= 0")
        if retrain_every <= 0:
            raise ValueError("retrain_every must be > 0")

        self.variable_order = list(variable_order)
        self.variable_bounds = dict(variable_bounds)
        self.pool_size = int(pool_size)
        self.simulator = simulator
        self.objective = objective
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.recorder = recorder
        self.budget = int(budget)
        self.retrain_every = int(retrain_every)
        self.n_init = int(n_init)
        self.seed = seed
        self.run_id = run_id
        self.checkpoint_path = Path(checkpoint_path)

        self._rng = np.random.default_rng(seed)
        self._history: list[dict[str, Any]] = []
        self._objective_cache: dict[str, float] = {}
        self._surrogate_ready = False
        self._engine_elapsed_total = 0.0
        self._sim_elapsed_total = 0.0

        candidates = self._generate_candidate_pool()
        self.state = OnlineOptimizationState(
            evaluated_geometries=[],
            candidate_geometries=[row.tolist() for row in candidates],
        )

        if resume:
            self.resume_from_checkpoint(self.checkpoint_path)

    def run(self) -> OnlineRunResult:
        self.recorder.start(
            {
                "run_id": self.run_id,
                "budget": self.budget,
                "retrain_every": self.retrain_every,
                "n_init": self.n_init,
                "seed": self.seed,
                "variable_order": self.variable_order,
                "variable_bounds": self.variable_bounds,
                "pool_size": self.pool_size,
                "simulator": self.simulator.__class__.__name__,
                "objective": self.objective.describe(),
                "surrogate": self.surrogate.__class__.__name__,
                "acquisition": self.acquisition.__class__.__name__,
                "checkpoint_path": str(self.checkpoint_path),
            }
        )

        for row in self._history:
            self.recorder.record_trial(row)

        if not self._history:
            self._initialize()

        while len(self._history) < (self.n_init + self.budget):
            if not self.state.candidate_geometries:
                break

            step = len(self._history)
            bo_step = max(step - self.n_init, 0)
            self.state.step = bo_step

            if step >= self.n_init and bo_step % self.retrain_every == 0:
                t0 = time.perf_counter()
                self._fit_surrogate()
                self._engine_elapsed_total += float(time.perf_counter() - t0)

            t_iter_start = time.perf_counter()
            t_engine_start = time.perf_counter()
            selected_geo, pred_mean, pred_uncertainty, acq_score = self._select_next_geometry(step)
            self._engine_elapsed_total += float(time.perf_counter() - t_engine_start)

            trial = self._evaluate_geometry(
                selected_geo,
                phase="bo" if step >= self.n_init else "init",
                iteration=step,
                predicted_mean=pred_mean,
                predicted_uncertainty=pred_uncertainty,
                acquisition_score=acq_score,
            )
            trial["t_iteration_sec"] = float(time.perf_counter() - t_iter_start)
            trial["t_engine_sec"] = max(0.0, float(trial["t_iteration_sec"]) - float(trial["t_simulator_sec"]))
            trial["t_engine_cum_sec"] = float(self._engine_elapsed_total)
            trial["t_simulator_cum_sec"] = float(self._sim_elapsed_total)
            self._record_trial(trial)
            self._save_checkpoint()

        if not self._history:
            raise RuntimeError("No trials were executed")

        best_trial = min(self._history, key=lambda row: float(row["objective"]))
        summary = {
            "num_trials": len(self._history),
            "best_objective": best_trial["objective"],
            "best_case_id": best_trial.get("case_id"),
            "total_engine_sec": float(self._engine_elapsed_total),
            "total_simulator_sec": float(self._sim_elapsed_total),
            "cache_hit_ratio": self._cache_hit_ratio(),
        }
        recorder_output = self.recorder.finalize(self.run_id, summary)

        return OnlineRunResult(
            run_id=self.run_id,
            best_trial=dict(best_trial),
            history=list(self._history),
            recorder_output=recorder_output,
            state=self.state,
            checkpoint_path=self.checkpoint_path,
        )

    def resume_from_checkpoint(self, path: str | Path) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            return
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        self.run_id = str(payload["run_id"])
        self.seed = payload.get("seed", self.seed)
        state = payload["state"]
        self.state = OnlineOptimizationState(
            evaluated_geometries=[list(map(float, row)) for row in state["evaluated_geometries"]],
            candidate_geometries=[list(map(float, row)) for row in state["candidate_geometries"]],
            best_geometry=None if state.get("best_geometry") is None else list(map(float, state["best_geometry"])),
            best_objective=state.get("best_objective"),
            step=int(state.get("step", 0)),
        )
        self._history = list(payload.get("history", []))
        self._objective_cache = dict(payload.get("objective_cache", {}))
        timing = payload.get("timing", {})
        self._engine_elapsed_total = float(timing.get("engine_elapsed_total", 0.0))
        self._sim_elapsed_total = float(timing.get("sim_elapsed_total", 0.0))
        if "rng_state" in payload:
            self._rng.bit_generator.state = payload["rng_state"]

    def _initialize(self) -> None:
        init_count = min(self.n_init, len(self.state.candidate_geometries))
        for iteration in range(init_count):
            selected = np.asarray(self.state.candidate_geometries[0], dtype=float)
            t_iter_start = time.perf_counter()
            trial = self._evaluate_geometry(
                selected,
                phase="init",
                iteration=iteration,
                predicted_mean=None,
                predicted_uncertainty=None,
                acquisition_score=None,
            )
            trial["t_iteration_sec"] = float(time.perf_counter() - t_iter_start)
            trial["t_engine_sec"] = max(0.0, float(trial["t_iteration_sec"]) - float(trial["t_simulator_sec"]))
            trial["t_engine_cum_sec"] = float(self._engine_elapsed_total)
            trial["t_simulator_cum_sec"] = float(self._sim_elapsed_total)
            self._record_trial(trial)
            self._save_checkpoint()

    def _fit_surrogate(self) -> None:
        if not self.state.evaluated_geometries:
            self._surrogate_ready = False
            return
        X_train = np.asarray(self.state.evaluated_geometries, dtype=float)
        y_train = np.asarray([float(row["objective"]) for row in self._history], dtype=float)
        try:
            self.surrogate.fit(X_train, y_train)
            self._surrogate_ready = True
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            self._surrogate_ready = False

    def _select_next_geometry(
        self, iteration: int
    ) -> tuple[np.ndarray, float | None, float | None, float | None]:
        if not self.state.candidate_geometries:
            raise RuntimeError("No candidate geometries left")

        candidates = np.asarray(self.state.candidate_geometries, dtype=float)
        if (not self._surrogate_ready) or len(self.state.evaluated_geometries) == 0:
            idx = int(self._rng.integers(0, len(candidates)))
            return candidates[idx], None, None, None

        try:
            pred = np.asarray(self.surrogate.predict(candidates), dtype=float).reshape(-1)
            unc_raw = self.surrogate.predict_uncertainty(candidates)
            unc = None if unc_raw is None else np.asarray(unc_raw, dtype=float).reshape(-1)
            scores = self.acquisition.score(pred, unc, best_y=self.state.best_objective)
            pos = int(np.argmax(scores))
            selected_unc = None if unc is None else float(unc[pos])
            return candidates[pos], float(pred[pos]), selected_unc, float(scores[pos])
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            idx = int(self._rng.integers(0, len(candidates)))
            return candidates[idx], None, None, None

    def _evaluate_geometry(
        self,
        geometry: np.ndarray,
        *,
        phase: str,
        iteration: int,
        predicted_mean: float | None,
        predicted_uncertainty: float | None,
        acquisition_score: float | None,
    ) -> dict[str, Any]:
        geometry = np.asarray(geometry, dtype=float).reshape(-1)

        if hasattr(self.simulator, "set_context"):
            try:
                self.simulator.set_context(run_id=self.run_id, iteration=iteration)  # type: ignore[attr-defined]
            except Exception:
                pass

        t_sim_start = time.perf_counter()
        simulation: SimulationResult = self.simulator.simulate(geometry)
        t_sim = float(time.perf_counter() - t_sim_start)
        self._sim_elapsed_total += t_sim

        if simulation.status != "done" or simulation.freqs is None or simulation.s_params is None:
            trial = {
                "iteration": int(iteration),
                "phase": phase,
                "geometry": geometry.tolist(),
                "metrics": {},
                "objective": float("inf"),
                "predicted_mean": predicted_mean,
                "predicted_uncertainty": predicted_uncertainty,
                "acquisition_score": acquisition_score,
                "best_so_far": self.state.best_objective,
                "cache_hit": self._metadata_value(simulation.metadata, "cache_hit"),
                "job_id": self._metadata_value(simulation.metadata, "job_id"),
                "case_id": self._metadata_value(simulation.metadata, "case_id"),
                "simulation_status": simulation.status,
                "simulation_error": simulation.error,
                "simulation_metadata": simulation.metadata,
                "t_engine_sec": 0.0,
                "t_simulator_sec": t_sim,
            }
            return trial

        objective_result = self.objective.evaluate(simulation.freqs, simulation.s_params)
        objective_value = float(objective_result.objective)

        self.state.evaluated_geometries.append(geometry.tolist())
        self.state.candidate_geometries = [
            row for row in self.state.candidate_geometries if not np.allclose(np.asarray(row, dtype=float), geometry)
        ]

        if self.state.best_objective is None or objective_value < float(self.state.best_objective):
            self.state.best_objective = objective_value
            self.state.best_geometry = geometry.tolist()

        trial = {
            "iteration": int(iteration),
            "phase": phase,
            "geometry": geometry.tolist(),
            "metrics": objective_result.metrics,
            "objective": objective_value,
            "predicted_mean": predicted_mean,
            "predicted_uncertainty": predicted_uncertainty,
            "acquisition_score": acquisition_score,
            "best_so_far": float(self.state.best_objective),
            "cache_hit": self._metadata_value(simulation.metadata, "cache_hit"),
            "job_id": self._metadata_value(simulation.metadata, "job_id"),
            "case_id": self._metadata_value(simulation.metadata, "case_id"),
            "simulation_status": simulation.status,
            "simulation_error": simulation.error,
            "simulation_metadata": simulation.metadata,
            "t_engine_sec": 0.0,
            "t_simulator_sec": t_sim,
        }
        return trial

    def _record_trial(self, trial: dict[str, Any]) -> None:
        self._history.append(dict(trial))
        self.recorder.record_trial(trial)

    def _generate_candidate_pool(self) -> np.ndarray:
        for name in self.variable_order:
            if name not in self.variable_bounds:
                raise ValueError(f"Missing bounds for variable {name}")
            low, high = self.variable_bounds[name]
            if float(low) >= float(high):
                raise ValueError(f"Invalid bounds for {name}: ({low}, {high})")
        rows = []
        for _ in range(self.pool_size):
            row = []
            for name in self.variable_order:
                low, high = self.variable_bounds[name]
                row.append(float(self._rng.uniform(low, high)))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def _cache_hit_ratio(self) -> float:
        if not self._history:
            return 0.0
        hits = 0
        for row in self._history:
            if bool(row.get("cache_hit", False)):
                hits += 1
        return float(hits / len(self._history))

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "seed": self.seed,
            "config": {
                "variable_order": self.variable_order,
                "variable_bounds": self.variable_bounds,
                "pool_size": self.pool_size,
                "budget": self.budget,
                "retrain_every": self.retrain_every,
                "n_init": self.n_init,
            },
            "state": {
                "evaluated_geometries": self.state.evaluated_geometries,
                "candidate_geometries": self.state.candidate_geometries,
                "best_geometry": self.state.best_geometry,
                "best_objective": self.state.best_objective,
                "step": self.state.step,
            },
            "history": self._history,
            "best": {
                "best_geometry": self.state.best_geometry,
                "best_objective": self.state.best_objective,
            },
            "timing": {
                "engine_elapsed_total": float(self._engine_elapsed_total),
                "sim_elapsed_total": float(self._sim_elapsed_total),
            },
            "sim_stats": {
                "num_calls": getattr(self.simulator, "num_calls", None),
                "num_success": getattr(self.simulator, "num_success", None),
                "num_failures": getattr(self.simulator, "num_failures", None),
            },
            "objective_cache": self._objective_cache,
            "rng_state": self._rng.bit_generator.state,
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    @staticmethod
    def _metadata_value(metadata: dict[str, Any] | None, key: str) -> Any:
        if not isinstance(metadata, dict):
            return None
        return metadata.get(key)
