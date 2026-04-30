from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from .candidate_space import CandidateSpace
from .optimization_spec import OptimizationSpec
from .recorder import ExperimentRecorder, RecorderOutput
from .simulators import SimulationResult, Simulator


@dataclass
class OptimizerState:
    evaluated_geometries: list[list[float]]
    evaluated_geometries_normalized: list[list[float]]
    best_geometry: list[float] | None = None
    best_objective: float | None = None
    step: int = 0


@dataclass(frozen=True)
class OptimizerRunResult:
    run_id: str
    best_trial: dict[str, Any]
    history: list[dict[str, Any]]
    recorder_output: RecorderOutput
    state: OptimizerState
    checkpoint_path: Path


@dataclass(frozen=True)
class SelectionResult:
    requested_geometry: np.ndarray
    predicted_mean: float | None = None
    predicted_uncertainty: float | None = None
    acquisition_score: float | None = None
    selection_debug: dict[str, Any] | None = None


@dataclass(frozen=True)
class SimulationRecord:
    selection: SelectionResult
    simulation: SimulationResult
    t_simulator_sec: float


class OptimizerBase(ABC):
    """Generic optimize-simulate-analyze loop for simulator-agnostic optimizers."""

    optimizer_backend = "general"

    def __init__(
        self,
        *,
        candidate_space: CandidateSpace,
        simulator: Simulator,
        optimization_spec: OptimizationSpec,
        recorder: ExperimentRecorder,
        budget: int,
        n_init: int = 0,
        seed: int | None = None,
        run_id: str = "online_optimizer",
        checkpoint_path: str | Path = "experiments/bo_online/checkpoint.json",
        resume: bool = False,
        progress_callback: Callable[[dict[str, Any], list[dict[str, Any]]], None] | None = None,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if n_init < 0:
            raise ValueError("n_init must be >= 0")

        self.candidate_space = candidate_space
        self.simulator = simulator
        self.optimization_spec = optimization_spec
        self.recorder = recorder
        self.budget = int(budget)
        self.n_init = int(n_init)
        self.seed = seed
        self.run_id = str(run_id)
        self.checkpoint_path = Path(checkpoint_path)
        self.progress_callback = progress_callback

        self._rng = np.random.default_rng(seed)
        self._history: list[dict[str, Any]] = []
        self._engine_elapsed_total = 0.0
        self._sim_elapsed_total = 0.0
        self.state = OptimizerState(evaluated_geometries=[], evaluated_geometries_normalized=[])

        if resume:
            self.resume_from_checkpoint(self.checkpoint_path)

    def run(self) -> OptimizerRunResult:
        self.on_run_start()
        target_trials = self.n_init + self.budget

        while len(self._history) < target_trials:
            if len(self.candidate_space.remaining()) == 0:
                break

            iteration = len(self._history)
            self.state.step = max(0, iteration - self.n_init)
            self.on_iteration_start(iteration)

            t_iter_start = time.perf_counter()
            t_engine_start = time.perf_counter()
            if iteration < self.n_init:
                selections = self._initial_selections()
            else:
                selections = self.plan_next(iteration)
            self._engine_elapsed_total += float(time.perf_counter() - t_engine_start)

            records = self.run_simulations(selections, iteration=iteration)

            t_engine_start = time.perf_counter()
            trials = self.collect_post_simulation_data(records, iteration=iteration)
            self.update_run_info(trials)
            self._engine_elapsed_total += float(time.perf_counter() - t_engine_start)

            for trial in trials:
                t_iteration = float(time.perf_counter() - t_iter_start)
                trial["t_iteration_sec"] = t_iteration
                trial["t_engine_sec"] = max(0.0, t_iteration - float(trial["t_simulator_sec"]))
                trial["t_engine_cum_sec"] = float(self._engine_elapsed_total)
                trial["t_simulator_cum_sec"] = float(self._sim_elapsed_total)
                self._record_trial(trial)

            self._save_checkpoint()
            self.on_iteration_end(iteration, trials)

        if not self._history:
            raise RuntimeError("No trials were executed")

        self.on_run_end()
        best_trial = min(self._history, key=lambda row: float(row["objective"]))
        summary = {
            "num_trials": len(self._history),
            "best_objective": best_trial["objective"],
            "best_case_id": best_trial.get("case_id"),
            "best_geometry": best_trial.get("geometry"),
            "total_engine_sec": float(self._engine_elapsed_total),
            "total_simulator_sec": float(self._sim_elapsed_total),
            "cache_hit_ratio": self._cache_hit_ratio(),
        }
        recorder_output = self.recorder.finalize(self.run_id, summary)
        return OptimizerRunResult(
            run_id=self.run_id,
            best_trial=dict(best_trial),
            history=list(self._history),
            recorder_output=recorder_output,
            state=self.state,
            checkpoint_path=self.checkpoint_path,
        )

    def on_run_start(self) -> None:
        self.recorder.start(
            {
                "run_id": self.run_id,
                "optimizer_backend": self.optimizer_backend,
                "budget": self.budget,
                "n_init": self.n_init,
                "seed": self.seed,
                "candidate_space": self.candidate_space.describe(),
                "simulator": self.simulator.__class__.__name__,
                "optimization_spec": self.optimization_spec.describe(),
                "checkpoint_path": str(self.checkpoint_path),
            }
        )
        for row in self._history:
            self.recorder.record_trial(row)

    def on_iteration_start(self, iteration: int) -> None:
        if hasattr(self.simulator, "set_context"):
            self.simulator.set_context(run_id=self.run_id, iteration=iteration)  # type: ignore[attr-defined]

    def on_iteration_end(self, iteration: int, trials: list[dict[str, Any]]) -> None:
        return None

    def on_run_end(self) -> None:
        return None

    @abstractmethod
    def plan_next(self, iteration: int) -> list[SelectionResult]:
        raise NotImplementedError

    def run_simulations(self, selections: Sequence[SelectionResult], *, iteration: int) -> list[SimulationRecord]:
        records: list[SimulationRecord] = []
        for selection in selections:
            t_sim_start = time.perf_counter()
            simulation = self.simulator.simulate(selection.requested_geometry)
            t_sim = float(time.perf_counter() - t_sim_start)
            self._sim_elapsed_total += t_sim
            records.append(SimulationRecord(selection=selection, simulation=simulation, t_simulator_sec=t_sim))
        return records

    def collect_post_simulation_data(
        self,
        records: Sequence[SimulationRecord],
        *,
        iteration: int,
    ) -> list[dict[str, Any]]:
        trials: list[dict[str, Any]] = []
        for record in records:
            trials.append(self._trial_from_record(record, iteration=iteration))
        return trials

    def update_run_info(self, trials: Sequence[dict[str, Any]]) -> None:
        for trial in trials:
            requested = np.asarray(trial["requested_geometry"], dtype=float)
            self.candidate_space.remove(requested)
            if trial["simulation_status"] != "done" or not np.isfinite(float(trial["objective"])):
                continue

            actual = np.asarray(trial["geometry"], dtype=float)
            self.candidate_space.remove(actual)
            self.state.evaluated_geometries.append(actual.tolist())
            self.state.evaluated_geometries_normalized.append(
                np.asarray(trial["geometry_normalized"], dtype=float).tolist()
            )
            objective = float(trial["objective"])
            if self.state.best_objective is None or objective < float(self.state.best_objective):
                self.state.best_objective = objective
                self.state.best_geometry = actual.tolist()

            trial["best_so_far"] = float(self.state.best_objective)

    def resume_from_checkpoint(self, path: str | Path) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            return
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        self.run_id = str(payload["run_id"])
        self.seed = payload.get("seed", self.seed)
        state = payload.get("state", {})
        self.state = OptimizerState(
            evaluated_geometries=[list(map(float, row)) for row in state.get("evaluated_geometries", [])],
            evaluated_geometries_normalized=[
                list(map(float, row)) for row in state.get("evaluated_geometries_normalized", [])
            ],
            best_geometry=None if state.get("best_geometry") is None else list(map(float, state["best_geometry"])),
            best_objective=state.get("best_objective"),
            step=int(state.get("step", 0)),
        )
        self._history = list(payload.get("history", []))
        if "candidate_space" in payload:
            self.candidate_space.load_state_dict(payload["candidate_space"])
        timing = payload.get("timing", {})
        self._engine_elapsed_total = float(timing.get("engine_elapsed_total", 0.0))
        self._sim_elapsed_total = float(timing.get("sim_elapsed_total", 0.0))
        if "rng_state" in payload:
            self._rng.bit_generator.state = payload["rng_state"]

    def _initial_selections(self) -> list[SelectionResult]:
        candidates = self.candidate_space.sample_initial(1, self._rng)
        if len(candidates) == 0:
            return []
        return [
            SelectionResult(
                requested_geometry=np.asarray(candidates[0], dtype=float),
                selection_debug={"mode": "init", "model_fit_status": "not_ready"},
            )
        ]

    def _trial_from_record(self, record: SimulationRecord, *, iteration: int) -> dict[str, Any]:
        selection = record.selection
        simulation = record.simulation
        requested = np.asarray(selection.requested_geometry, dtype=float).reshape(-1)
        actual = self._actual_geometry(requested, simulation)
        actual_normalized = self.candidate_space.normalize(actual)
        selection_debug = dict(selection.selection_debug or {})
        if simulation.actual_geometry is None:
            selection_debug["actual_geometry_source"] = "requested_geometry_fallback"
        else:
            selection_debug["actual_geometry_source"] = "simulation.actual_geometry"

        if simulation.status != "done" or simulation.freqs is None or simulation.s_params is None:
            spec_results = None
            objective_value = float("inf")
            objective_array = np.asarray([float("inf")], dtype=float)
            constraints_array = np.asarray([], dtype=float)
            metrics: dict[str, float] = {}
            objective_components: dict[str, float] = {}
            metadata: dict[str, Any] = {}
        else:
            spec_results = self.optimization_spec.evaluate(simulation.freqs, simulation.s_params)
            objective_value = spec_results.scalar_objective()
            objective_array = spec_results.objective
            constraints_array = spec_results.constraints
            metrics = spec_results.metrics
            objective_components = spec_results.objective_components
            metadata = spec_results.metadata

        best_so_far = self.state.best_objective
        if np.isfinite(objective_value):
            best_so_far = objective_value if best_so_far is None else min(float(best_so_far), objective_value)

        return {
            "iteration": int(iteration),
            "step": self.state.step,
            "phase": "init" if iteration < self.n_init else "bo",
            "requested_geometry": requested.tolist(),
            "geometry": actual.tolist(),
            "geometry_normalized": np.asarray(actual_normalized, dtype=float).reshape(-1).tolist(),
            "objective": objective_value,
            "objective_array": objective_array.tolist(),
            "constraints_array": constraints_array.tolist(),
            "metrics": metrics,
            "objective_components": objective_components,
            "constraint_metrics": dict(metadata.get("constraint_metrics", {})),
            "constraint_violations": dict(metadata.get("constraint_violations", {})),
            "feasible": metadata.get("feasible"),
            "constraint_margin_min": metadata.get("constraint_margin_min"),
            "predicted_mean": selection.predicted_mean,
            "predicted_uncertainty": selection.predicted_uncertainty,
            "acquisition_score": selection.acquisition_score,
            "best_so_far": best_so_far,
            "cache_hit": self._metadata_value(simulation.metadata, "cache_hit"),
            "job_id": self._metadata_value(simulation.metadata, "job_id"),
            "case_id": self._metadata_value(simulation.metadata, "case_id"),
            "simulation_status": simulation.status,
            "simulation_error": simulation.error,
            "simulation_metadata": simulation.metadata,
            "t_engine_sec": 0.0,
            "t_simulator_sec": record.t_simulator_sec,
            "optimizer_backend": self.optimizer_backend,
            "selection_debug": selection_debug,
            "spec_results_metadata": metadata,
        }

    def _actual_geometry(self, requested: np.ndarray, simulation: SimulationResult) -> np.ndarray:
        if simulation.actual_geometry is None:
            return np.asarray(requested, dtype=float).reshape(-1)
        return np.asarray(simulation.actual_geometry, dtype=float).reshape(-1)

    def _record_trial(self, trial: dict[str, Any]) -> None:
        self._history.append(dict(trial))
        self.recorder.record_trial(trial)
        if self.progress_callback is not None:
            self.progress_callback(dict(trial), list(self._history))

    def _cache_hit_ratio(self) -> float:
        if not self._history:
            return 0.0
        hits = sum(1 for row in self._history if bool(row.get("cache_hit", False)))
        return float(hits / len(self._history))

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "seed": self.seed,
            "optimizer_backend": self.optimizer_backend,
            "config": {
                "budget": self.budget,
                "n_init": self.n_init,
                "candidate_space": self.candidate_space.describe(),
            },
            "state": {
                "evaluated_geometries": self.state.evaluated_geometries,
                "evaluated_geometries_normalized": self.state.evaluated_geometries_normalized,
                "best_geometry": self.state.best_geometry,
                "best_objective": self.state.best_objective,
                "step": self.state.step,
            },
            "candidate_space": self.candidate_space.state_dict(),
            "history": self._history,
            "timing": {
                "engine_elapsed_total": float(self._engine_elapsed_total),
                "sim_elapsed_total": float(self._sim_elapsed_total),
            },
            "sim_stats": {
                "num_calls": getattr(self.simulator, "num_calls", None),
                "num_success": getattr(self.simulator, "num_success", None),
                "num_failures": getattr(self.simulator, "num_failures", None),
            },
            "rng_state": self._rng.bit_generator.state,
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    @staticmethod
    def _metadata_value(metadata: dict[str, Any] | None, key: str) -> Any:
        if not isinstance(metadata, dict):
            return None
        return metadata.get(key)


# Backward-compatible aliases for older imports.
OnlineOptimizationState = OptimizerState
OnlineRunResult = OptimizerRunResult
