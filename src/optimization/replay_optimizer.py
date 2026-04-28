from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .acquisition import AcquisitionFunction
from .datasets import PassiveDataset
from .objectives import Objective
from .recorder import ExperimentRecorder, RecorderOutput
from .simulators import Simulator
from .surrogates import SurrogateModel


@dataclass
class OptimizationState:
    train_indices: list[int]
    candidate_indices: list[int]
    test_indices: list[int]
    best_index: int | None = None
    best_objective: float | None = None
    step: int = 0


@dataclass(frozen=True)
class RunResult:
    run_id: str
    best_trial: dict[str, Any]
    history: list[dict[str, Any]]
    recorder_output: RecorderOutput
    state: OptimizationState


class BaseOptimizer(ABC):
    @abstractmethod
    def run(self) -> RunResult:
        raise NotImplementedError


class ReplayOptimizer(BaseOptimizer):
    """Controller that orchestrates replay BO loop via dataset/simulator/objective/model/acquisition."""

    def __init__(
        self,
        *,
        dataset: PassiveDataset,
        simulator: Simulator,
        objective: Objective,
        surrogate: SurrogateModel,
        acquisition: AcquisitionFunction,
        recorder: ExperimentRecorder,
        initial_indices: Sequence[int],
        candidate_indices: Sequence[int],
        budget: int,
        retrain_every: int = 1,
        n_init: int | None = None,
        seed: int | None = None,
        test_indices: Sequence[int] | None = None,
        run_id: str = "bo_replay",
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if retrain_every <= 0:
            raise ValueError("retrain_every must be > 0")
        if n_init is not None and n_init < 0:
            raise ValueError("n_init must be >= 0")

        self.dataset = dataset
        self.simulator = simulator
        self.objective = objective
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.recorder = recorder

        self.initial_indices = self._normalize_indices(initial_indices)
        self.budget = int(budget)
        self.retrain_every = int(retrain_every)
        self.n_init = n_init
        self.seed = seed
        self.run_id = run_id

        candidates = self._normalize_indices(candidate_indices)
        tests = self._normalize_indices(test_indices or [])
        self.state = OptimizationState(train_indices=[], candidate_indices=candidates, test_indices=tests)

        self._rng = np.random.default_rng(self.seed)
        self._history: list[dict[str, Any]] = []
        self._objective_cache: dict[int, float] = {}
        self._surrogate_ready = False

    def run(self) -> RunResult:
        self.recorder.start(
            {
                "run_id": self.run_id,
                "budget": self.budget,
                "retrain_every": self.retrain_every,
                "n_init": self.n_init,
                "seed": self.seed,
                "dataset_size": len(self.dataset),
                "simulator": self.simulator.__class__.__name__,
                "objective": self.objective.describe(),
                "surrogate": self.surrogate.__class__.__name__,
                "acquisition": self.acquisition.__class__.__name__,
                "initial_indices": list(self.initial_indices),
                "candidate_indices": list(self.state.candidate_indices),
                "test_indices": list(self.state.test_indices),
            }
        )

        self._initialize()

        for step in range(self.budget):
            if not self.state.candidate_indices:
                break

            self.state.step = step
            if step % self.retrain_every == 0:
                self._fit_surrogate()

            selected_idx, pred_mean, pred_uncertainty, acq_score = self._select_next_index()
            trial = self._evaluate_index(
                selected_idx,
                phase="bo",
                predicted_mean=pred_mean,
                predicted_uncertainty=pred_uncertainty,
                acquisition_score=acq_score,
            )
            self._record_trial(trial)

        if not self._history:
            raise RuntimeError("No trials were executed; check initialization and budget")

        best_trial = min(self._history, key=lambda row: float(row["objective"]))
        summary = {
            "num_trials": len(self._history),
            "best_objective": best_trial["objective"],
            "best_dataset_index": best_trial["dataset_index"],
            "best_case_id": best_trial["case_id"],
        }
        recorder_output = self.recorder.finalize(self.run_id, summary)

        return RunResult(
            run_id=self.run_id,
            best_trial=dict(best_trial),
            history=list(self._history),
            recorder_output=recorder_output,
            state=self.state,
        )

    def _initialize(self) -> None:
        init_indices = list(self.initial_indices)
        if self.n_init is not None and len(init_indices) < self.n_init:
            available = [idx for idx in self.state.candidate_indices if idx not in init_indices]
            extra_count = min(self.n_init - len(init_indices), len(available))
            if extra_count > 0:
                extra = self._rng.choice(available, size=extra_count, replace=False).tolist()
                init_indices.extend(int(i) for i in extra)

        for idx in init_indices:
            trial = self._evaluate_index(
                idx,
                phase="init",
                predicted_mean=None,
                predicted_uncertainty=None,
                acquisition_score=None,
            )
            self._record_trial(trial)

    def _fit_surrogate(self) -> None:
        if not self.state.train_indices:
            self._surrogate_ready = False
            return

        X_train = np.asarray([self.dataset.get_geometry(i) for i in self.state.train_indices], dtype=float)
        y_train = np.asarray([self._objective_cache[i] for i in self.state.train_indices], dtype=float)

        try:
            self.surrogate.fit(X_train, y_train)
            self._surrogate_ready = True
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            self._surrogate_ready = False

    def _select_next_index(self) -> tuple[int, float | None, float | None, float | None]:
        candidates = list(self.state.candidate_indices)
        if not candidates:
            raise RuntimeError("No candidates left to select")

        if not self._surrogate_ready or not self.state.train_indices:
            selected = int(self._rng.choice(candidates))
            return selected, None, None, None

        X_pool = np.asarray([self.dataset.get_geometry(i) for i in candidates], dtype=float)
        try:
            pred = np.asarray(self.surrogate.predict(X_pool), dtype=float).reshape(-1)
            unc_raw = self.surrogate.predict_uncertainty(X_pool)
            unc = None if unc_raw is None else np.asarray(unc_raw, dtype=float).reshape(-1)
            best_y = self.state.best_objective
            scores = self.acquisition.score(pred, unc, best_y=best_y)

            pos = int(np.argmax(scores))
            selected_idx = int(candidates[pos])
            selected_unc = None if unc is None else float(unc[pos])
            return selected_idx, float(pred[pos]), selected_unc, float(scores[pos])
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            selected = int(self._rng.choice(candidates))
            return selected, None, None, None

    def _evaluate_index(
        self,
        index: int,
        *,
        phase: str,
        predicted_mean: float | None,
        predicted_uncertainty: float | None,
        acquisition_score: float | None,
    ) -> dict[str, Any]:
        if index not in self.state.candidate_indices and index not in self.state.train_indices:
            raise ValueError(f"Index {index} is not part of known optimization pools")

        geometry = self.dataset.get_geometry(index)
        simulation = self.simulator.simulate(geometry)
        if simulation.status != "done" or simulation.freqs is None or simulation.s_params is None:
            raise RuntimeError(simulation.error or f"Simulation failed at index {index}")

        objective_result = self.objective.evaluate(simulation.freqs, simulation.s_params)
        objective_value = float(objective_result.objective)
        self._objective_cache[index] = objective_value

        if index not in self.state.train_indices:
            self.state.train_indices.append(index)
        self.state.candidate_indices = [i for i in self.state.candidate_indices if i != index]

        if self.state.best_objective is None or objective_value < self.state.best_objective:
            self.state.best_objective = objective_value
            self.state.best_index = index

        trial = {
            "iteration": len(self._history),
            "step": self.state.step,
            "phase": phase,
            "dataset_index": index,
            "case_id": self.dataset.get_case_id(index),
            "geo_params": geometry,
            "metrics": objective_result.metrics,
            "objective": objective_value,
            "predicted_mean": predicted_mean,
            "predicted_uncertainty": predicted_uncertainty,
            "acquisition_score": acquisition_score,
            "best_so_far": float(self.state.best_objective),
            "simulation_status": simulation.status,
            "simulation_error": simulation.error,
            "simulation_metadata": simulation.metadata,
        }
        return trial

    def _record_trial(self, trial: dict[str, Any]) -> None:
        self._history.append(trial)
        self.recorder.record_trial(trial)

    def _normalize_indices(self, indices: Sequence[int]) -> list[int]:
        normalized = sorted(set(int(i) for i in indices))
        for idx in normalized:
            if idx < 0 or idx >= len(self.dataset):
                raise IndexError(f"Dataset index out of range: {idx}")
        return normalized
