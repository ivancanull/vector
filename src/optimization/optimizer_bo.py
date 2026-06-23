from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .botorch_acquisition import BoTorchAcquisitionBackend, GPqLogEIAcquisitionBackend
from .candidate_space import ContinuousCandidateSpace
from .optimizer_base import OptimizerBase, SelectionResult, SimulationRecord


class OptimizerBoTorch(OptimizerBase):
    """BoTorch single-objective optimizer over a CandidateSpace."""

    optimizer_backend = "botorch"

    def __init__(
        self,
        *,
        acquisition_backend: BoTorchAcquisitionBackend | None = None,
        retrain_every: int = 1,
        use_trust_region: bool = False,
        tr_length_init: float = 0.25,
        tr_length_min: float = 0.03,
        tr_length_max: float = 1.0,
        tr_success_tolerance: int = 3,
        tr_failure_tolerance: int = 6,
        tr_expand_factor: float = 2.0,
        tr_shrink_factor: float = 0.5,
        tr_improvement_epsilon: float = 1e-9,
        not_ready_selection_policy: str = "first",
        **kwargs: Any,
    ) -> None:
        if retrain_every <= 0:
            raise ValueError("retrain_every must be > 0")
        if tr_length_init <= 0 or tr_length_min <= 0 or tr_length_max <= 0:
            raise ValueError("trust-region lengths must be > 0")
        if tr_length_min > tr_length_init or tr_length_init > tr_length_max:
            raise ValueError("require tr_length_min <= tr_length_init <= tr_length_max")
        if tr_success_tolerance <= 0 or tr_failure_tolerance <= 0:
            raise ValueError("trust-region success/failure tolerances must be > 0")
        if tr_expand_factor <= 1.0:
            raise ValueError("tr_expand_factor must be > 1.0")
        if not (0.0 < tr_shrink_factor < 1.0):
            raise ValueError("tr_shrink_factor must be in (0, 1)")
        policy = str(not_ready_selection_policy).strip().lower()
        if policy not in {"first", "random"}:
            raise ValueError("not_ready_selection_policy must be 'first' or 'random'")
        self.acquisition_backend = acquisition_backend or GPqLogEIAcquisitionBackend()
        self.retrain_every = int(retrain_every)
        self.use_trust_region = bool(use_trust_region)
        self.tr_length_init = float(tr_length_init)
        self.tr_length_min = float(tr_length_min)
        self.tr_length_max = float(tr_length_max)
        self.tr_success_tolerance = int(tr_success_tolerance)
        self.tr_failure_tolerance = int(tr_failure_tolerance)
        self.tr_expand_factor = float(tr_expand_factor)
        self.tr_shrink_factor = float(tr_shrink_factor)
        self.tr_improvement_epsilon = float(tr_improvement_epsilon)
        self.not_ready_selection_policy = policy
        self._model_ready = False
        self._last_fit_error: str | None = None
        self._tr_center_norm: np.ndarray | None = None
        self._tr_length = float(tr_length_init)
        self._tr_success_count = 0
        self._tr_failure_count = 0
        self._tr_best_objective: float | None = None
        self._last_tr_update_action = "init"
        super().__init__(**kwargs)

    def plan_next(self, iteration: int) -> list[SelectionResult]:
        self._update_trust_region_state()
        if self.state.step % self.retrain_every == 0:
            self._fit_model()
        selection = self._select_next()
        return [selection]

    def _fit_model(self) -> None:
        valid_trials = [
            row
            for row in self._history
            if row.get("simulation_status") == "done" and np.isfinite(float(row.get("objective", float("inf"))))
            and not bool(row.get("exclude_from_model_training", False))
        ]
        if not valid_trials:
            self._model_ready = False
            self._last_fit_error = None
            return

        X_actual_np = np.asarray([row["geometry"] for row in valid_trials], dtype=float)
        X_np = np.asarray([row["geometry_normalized"] for row in valid_trials], dtype=float)
        y_np = np.asarray([self._extract_scalar_objective(row) for row in valid_trials], dtype=float).reshape(-1, 1)
        result = self.acquisition_backend.fit(
            X_np,
            y_np,
            seed=self.seed,
            X_actual=X_actual_np,
            history=valid_trials,
            optimization_spec=self.optimization_spec,
            variable_order=self.candidate_space.variable_order,
        )
        self._model_ready = bool(result.ready)
        self._last_fit_error = result.error

    def _select_next(self, candidates_actual: np.ndarray | None = None) -> SelectionResult:
        if candidates_actual is None:
            continuous_selection = self._select_next_continuous()
            if continuous_selection is not None:
                return continuous_selection
            resampled_selection = self._select_next_resampled_continuous()
            if resampled_selection is not None:
                return resampled_selection
            candidates_actual = self.candidate_space.remaining()
        candidates_actual = np.asarray(candidates_actual, dtype=float)
        if len(candidates_actual) == 0:
            raise RuntimeError("No candidates left to select")
        candidates_actual, tr_debug = self._apply_trust_region_filter(candidates_actual)

        if not self._model_ready:
            if self.not_ready_selection_policy == "random":
                return self._random_selection(
                    candidates_actual,
                    {
                        "mode": "random_not_ready",
                        "model_fit_status": "failed" if self._last_fit_error else "not_ready",
                        "fit_error": self._last_fit_error,
                        "fallback_policy": "random_remaining_candidate",
                        **tr_debug,
                    },
                )
            return self._deterministic_selection(
                candidates_actual,
                {
                    "mode": "deterministic_not_ready",
                    "model_fit_status": "failed" if self._last_fit_error else "not_ready",
                    "fit_error": self._last_fit_error,
                    "fallback_policy": "first_remaining_candidate",
                    **tr_debug,
                },
            )

        try:
            return self.acquisition_backend.select(
                candidates_actual=candidates_actual,
                candidates_normalized=self.candidate_space.normalize(candidates_actual),
                history=self._history,
                tr_debug=tr_debug,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return self._deterministic_selection(
                candidates_actual,
                {
                    "mode": "deterministic_select_error",
                    "model_fit_status": "failed",
                    "fit_error": self._last_fit_error,
                    "select_error": f"{type(exc).__name__}: {exc}",
                    "fallback_policy": "first_remaining_candidate",
                    **tr_debug,
                },
            )

    def _select_next_continuous(self) -> SelectionResult | None:
        if not isinstance(self.candidate_space, ContinuousCandidateSpace):
            return None
        select_continuous = getattr(self.acquisition_backend, "select_continuous", None)
        if not callable(select_continuous):
            return None
        if not self._model_ready:
            return None

        bounds_normalized, tr_debug = self._continuous_normalized_bounds()
        try:
            return select_continuous(
                bounds_normalized=bounds_normalized,
                candidate_space=self.candidate_space,
                history=self._history,
                tr_debug=tr_debug,
                evaluated_actual=np.asarray(self.state.evaluated_geometries, dtype=float),
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return self._random_continuous_selection(
                bounds_normalized,
                {
                    "mode": "random_continuous_select_error",
                    "model_fit_status": "failed",
                    "fit_error": self._last_fit_error,
                    "select_error": f"{type(exc).__name__}: {exc}",
                    "fallback_policy": "random_continuous_candidate",
                    **tr_debug,
                },
            )

    def _select_next_resampled_continuous(self) -> SelectionResult | None:
        if not isinstance(self.candidate_space, ContinuousCandidateSpace):
            return None

        bounds_normalized, tr_debug = self._continuous_normalized_bounds()
        if not self._model_ready:
            return self._random_continuous_selection(
                bounds_normalized,
                {
                    "mode": "random_not_ready_continuous",
                    "model_fit_status": "failed" if self._last_fit_error else "not_ready",
                    "fit_error": self._last_fit_error,
                    "fallback_policy": "random_continuous_candidate",
                    **tr_debug,
                },
            )

        candidates_actual, sample_debug = self._resampled_continuous_candidates(bounds_normalized)
        if len(candidates_actual) == 0:
            return self._random_continuous_selection(
                bounds_normalized,
                {
                    "mode": "random_continuous_resample_empty",
                    "model_fit_status": "ok",
                    "fallback_policy": "random_continuous_candidate",
                    **tr_debug,
                    **sample_debug,
                },
            )

        try:
            return self.acquisition_backend.select(
                candidates_actual=candidates_actual,
                candidates_normalized=self.candidate_space.normalize(candidates_actual),
                history=self._history,
                tr_debug={**tr_debug, **sample_debug},
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return self._random_continuous_selection(
                bounds_normalized,
                {
                    "mode": "random_continuous_select_error",
                    "model_fit_status": "failed",
                    "fit_error": self._last_fit_error,
                    "select_error": f"{type(exc).__name__}: {exc}",
                    "fallback_policy": "random_continuous_candidate",
                    **tr_debug,
                    **sample_debug,
                },
            )

    def _resampled_continuous_candidates(self, bounds_normalized: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(self.candidate_space, ContinuousCandidateSpace):
            raise RuntimeError("continuous resampling requires ContinuousCandidateSpace")
        target_count = int(getattr(self.candidate_space, "pool_size", 0))
        if target_count <= 0:
            raise ValueError("continuous resampling requires candidate_space.pool_size > 0")

        seen = self._evaluated_geometry_keys()
        rows: list[np.ndarray] = []
        attempts = 0
        raw_count = 0
        duplicate_count = 0
        max_attempts = 5
        while len(rows) < target_count and attempts < max_attempts:
            attempts += 1
            batch = self.candidate_space.sample_uniform(
                target_count,
                self._rng,
                bounds_normalized=bounds_normalized,
            )
            raw_count += int(len(batch))
            for row in np.asarray(batch, dtype=float):
                key = tuple(np.round(row.reshape(-1), 8))
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                rows.append(row.reshape(-1))
                if len(rows) >= target_count:
                    break

        candidates = (
            np.asarray(rows, dtype=float)
            if rows
            else np.empty((0, len(self.candidate_space.variable_order)), dtype=float)
        )
        return candidates, {
            "candidate_source": "resampled_continuous_bounds",
            "resampled_candidate_target": int(target_count),
            "resampled_candidate_count": int(len(candidates)),
            "resampled_attempts": int(attempts),
            "resampled_raw_count": int(raw_count),
            "resampled_duplicate_count": int(duplicate_count),
        }

    def _evaluated_geometry_keys(self) -> set[tuple[float, ...]]:
        if not self.state.evaluated_geometries:
            return set()
        evaluated = np.asarray(self.state.evaluated_geometries, dtype=float).reshape(
            -1,
            len(self.candidate_space.variable_order),
        )
        return {tuple(np.round(row.reshape(-1), 8)) for row in evaluated}

    def _continuous_normalized_bounds(self) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(self.candidate_space, ContinuousCandidateSpace):
            raise RuntimeError("continuous bounds require ContinuousCandidateSpace")
        base_bounds = self.candidate_space.normalized_bounds()
        if not self.use_trust_region:
            return base_bounds, {"tr_active": False, "candidate_source": "continuous_acqf"}

        center = self._resolve_tr_center_norm()
        if center is None:
            return base_bounds, {
                "tr_active": True,
                "tr_filter_applied": False,
                "tr_filter_reason": "center_unavailable",
                "tr_length": float(self._tr_length),
                "candidate_source": "continuous_acqf_global_fallback",
            }

        lower = np.maximum(base_bounds[0], center.reshape(-1) - float(self._tr_length))
        upper = np.minimum(base_bounds[1], center.reshape(-1) + float(self._tr_length))
        if np.any(upper <= lower):
            return base_bounds, {
                "tr_active": True,
                "tr_filter_applied": False,
                "tr_filter_reason": "empty_region_fallback_global",
                "tr_length": float(self._tr_length),
                "tr_center": center.tolist(),
                "tr_update_action": self._last_tr_update_action,
                "candidate_source": "continuous_acqf_global_fallback",
            }
        return np.vstack([lower, upper]), {
            "tr_active": True,
            "tr_filter_applied": True,
            "tr_length": float(self._tr_length),
            "tr_center": center.tolist(),
            "tr_update_action": self._last_tr_update_action,
            "candidate_source": "continuous_acqf_trust_region",
        }

    def _random_continuous_selection(self, bounds_normalized: np.ndarray, debug: dict[str, Any]) -> SelectionResult:
        if not isinstance(self.candidate_space, ContinuousCandidateSpace):
            raise RuntimeError("random continuous selection requires ContinuousCandidateSpace")
        bounds = np.asarray(bounds_normalized, dtype=float)
        if bounds.shape != (2, len(self.candidate_space.variable_order)):
            raise ValueError("bounds_normalized shape mismatch")
        evaluated = np.asarray(self.state.evaluated_geometries, dtype=float)
        max_attempts = 100
        selected_actual: np.ndarray | None = None
        for attempt in range(max_attempts):
            selected_norm = self._rng.uniform(bounds[0], bounds[1])
            actual = np.asarray(self.candidate_space.denormalize(selected_norm), dtype=float).reshape(-1)
            actual = np.asarray(self.candidate_space.project(actual), dtype=float).reshape(-1)
            if not self._is_evaluated_geometry(actual, evaluated):
                selected_actual = actual
                debug = dict(debug)
                debug.update(
                    {
                        "random_continuous_attempts": int(attempt + 1),
                        "selected_normalized": self.candidate_space.normalize(actual).tolist(),
                    }
                )
                break
        if selected_actual is None:
            selected_norm = self._rng.uniform(bounds[0], bounds[1])
            selected_actual = np.asarray(
                self.candidate_space.project(self.candidate_space.denormalize(selected_norm)),
                dtype=float,
            ).reshape(-1)
            debug = dict(debug)
            debug.update(
                {
                    "random_continuous_attempts": int(max_attempts),
                    "selected_normalized": self.candidate_space.normalize(selected_actual).tolist(),
                    "duplicate_avoidance_exhausted": True,
                }
            )
        return SelectionResult(
            requested_geometry=selected_actual,
            selection_debug=debug,
        )

    def _is_evaluated_geometry(self, actual: np.ndarray, evaluated: np.ndarray) -> bool:
        if evaluated.size == 0:
            return False
        evaluated = evaluated.reshape(-1, len(np.asarray(actual).reshape(-1)))
        target = np.asarray(actual, dtype=float).reshape(1, -1)
        return bool(
            np.any(
                np.all(
                    np.isclose(
                        evaluated,
                        target,
                        atol=float(getattr(self.candidate_space, "match_atol", 1e-9)),
                        rtol=float(getattr(self.candidate_space, "match_rtol", 1e-9)),
                    ),
                    axis=1,
                )
            )
        )

    def _deterministic_selection(self, candidates_actual: np.ndarray, debug: dict[str, Any]) -> SelectionResult:
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[0], dtype=float),
            selection_debug=debug,
        )

    def _random_selection(self, candidates_actual: np.ndarray, debug: dict[str, Any]) -> SelectionResult:
        pos = int(self._rng.integers(0, len(candidates_actual)))
        debug = dict(debug)
        debug.update(
            {
                "selected_pos": pos,
                "candidate_pool_size": int(len(candidates_actual)),
            }
        )
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[pos], dtype=float),
            selection_debug=debug,
        )

    def _apply_trust_region_filter(self, candidates_actual: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.use_trust_region:
            return candidates_actual, {"tr_active": False}
        center = self._resolve_tr_center_norm()
        if center is None:
            return candidates_actual, {
                "tr_active": True,
                "tr_filter_applied": False,
                "tr_filter_reason": "center_unavailable",
                "tr_length": float(self._tr_length),
            }

        Xcand_np = self.candidate_space.normalize(candidates_actual)
        d_inf = np.max(np.abs(Xcand_np - center.reshape(1, -1)), axis=1)
        inside = d_inf <= float(self._tr_length)
        tr_count = int(np.count_nonzero(inside))
        if tr_count == 0:
            return candidates_actual, {
                "tr_active": True,
                "tr_filter_applied": False,
                "tr_filter_reason": "empty_region_fallback_global",
                "tr_length": float(self._tr_length),
                "tr_candidate_count": int(len(candidates_actual)),
                "tr_candidates_in_region": 0,
                "tr_center": center.tolist(),
                "tr_update_action": self._last_tr_update_action,
            }
        filtered = candidates_actual[inside]
        return filtered, {
            "tr_active": True,
            "tr_filter_applied": True,
            "tr_length": float(self._tr_length),
            "tr_candidate_count": int(len(candidates_actual)),
            "tr_candidates_in_region": tr_count,
            "tr_center": center.tolist(),
            "tr_update_action": self._last_tr_update_action,
        }

    def _resolve_tr_center_norm(self) -> np.ndarray | None:
        if self._tr_center_norm is not None:
            return np.asarray(self._tr_center_norm, dtype=float).reshape(-1)
        if self.state.best_geometry is None:
            return None
        self._tr_center_norm = np.asarray(
            self.candidate_space.normalize(np.asarray(self.state.best_geometry, dtype=float)),
            dtype=float,
        ).reshape(-1)
        return np.asarray(self._tr_center_norm, dtype=float).reshape(-1)

    def _update_trust_region_state(self) -> None:
        if not self.use_trust_region:
            return
        if self.state.best_geometry is not None:
            self._tr_center_norm = np.asarray(
                self.candidate_space.normalize(np.asarray(self.state.best_geometry, dtype=float)),
                dtype=float,
            ).reshape(-1)
        current_best = self.state.best_objective
        if current_best is None:
            self._last_tr_update_action = "wait_for_best"
            return
        if self._tr_best_objective is None:
            self._tr_best_objective = float(current_best)
            self._tr_success_count = 0
            self._tr_failure_count = 0
            self._last_tr_update_action = "init_best"
            return

        improved = float(current_best) < float(self._tr_best_objective) - self.tr_improvement_epsilon
        if improved:
            self._tr_best_objective = float(current_best)
            self._tr_success_count += 1
            self._tr_failure_count = 0
            if self._tr_success_count >= self.tr_success_tolerance:
                self._tr_length = min(self.tr_length_max, self._tr_length * self.tr_expand_factor)
                self._tr_success_count = 0
                self._last_tr_update_action = "expand"
            else:
                self._last_tr_update_action = "success_hold"
            return

        self._tr_failure_count += 1
        self._tr_success_count = 0
        if self._tr_failure_count >= self.tr_failure_tolerance:
            new_len = self._tr_length * self.tr_shrink_factor
            if new_len < self.tr_length_min:
                self._tr_length = self.tr_length_min
                # Restart center on current incumbent while keeping the run alive.
                if self.state.best_geometry is not None:
                    self._tr_center_norm = np.asarray(
                        self.candidate_space.normalize(np.asarray(self.state.best_geometry, dtype=float)),
                        dtype=float,
                    ).reshape(-1)
                self._last_tr_update_action = "restart_at_min"
            else:
                self._tr_length = max(self.tr_length_min, new_len)
                self._last_tr_update_action = "shrink"
            self._tr_failure_count = 0
        else:
            self._last_tr_update_action = "failure_hold"

    def resume_from_checkpoint(self, path: str | Path) -> None:
        super().resume_from_checkpoint(path)
        try:
            import json
            from pathlib import Path

            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return
        tr = payload.get("trust_region", {})
        if not isinstance(tr, dict):
            return
        center = tr.get("center_norm")
        self._tr_center_norm = None if center is None else np.asarray(center, dtype=float).reshape(-1)
        self._tr_length = float(tr.get("length", self._tr_length))
        self._tr_success_count = int(tr.get("success_count", 0))
        self._tr_failure_count = int(tr.get("failure_count", 0))
        best = tr.get("best_objective")
        self._tr_best_objective = None if best is None else float(best)
        self._last_tr_update_action = str(tr.get("last_action", self._last_tr_update_action))

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        try:
            import json

            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return
        payload["trust_region"] = {
            "enabled": self.use_trust_region,
            "center_norm": None if self._tr_center_norm is None else np.asarray(self._tr_center_norm, dtype=float).reshape(-1).tolist(),
            "length": float(self._tr_length),
            "success_count": int(self._tr_success_count),
            "failure_count": int(self._tr_failure_count),
            "best_objective": self._tr_best_objective,
            "last_action": self._last_tr_update_action,
            "config": {
                "length_init": float(self.tr_length_init),
                "length_min": float(self.tr_length_min),
                "length_max": float(self.tr_length_max),
                "success_tolerance": int(self.tr_success_tolerance),
                "failure_tolerance": int(self.tr_failure_tolerance),
                "expand_factor": float(self.tr_expand_factor),
                "shrink_factor": float(self.tr_shrink_factor),
                "improvement_epsilon": float(self.tr_improvement_epsilon),
            },
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    @staticmethod
    def _extract_scalar_objective(row: dict[str, Any]) -> float:
        arr = np.asarray(row.get("objective_array", []), dtype=float).reshape(-1)
        if arr.shape != (1,):
            raise ValueError(f"Only scalar objective arrays are supported now; got shape {arr.shape}")
        return float(arr[0])


BoTorchOptimizer = OptimizerBoTorch
OnlineBOOptimizer = OptimizerBoTorch


class OWAFSBNNOptimizer(OptimizerBoTorch):
    """BNN optimizer that evaluates post-init proposals with OWAFS + VF."""

    optimizer_backend = "owafs_bnn"

    def run_simulations(self, selections: list[SelectionResult], *, iteration: int) -> list[SimulationRecord]:
        records: list[SimulationRecord] = []
        for selection in selections:
            t_sim_start = time.perf_counter()
            if iteration < self.n_init:
                simulation = self.simulator.simulate(selection.requested_geometry)
            else:
                predict_profile = getattr(self.acquisition_backend, "predict_response_profile", None)
                simulate_afs = getattr(self.simulator, "simulate_afs", None)
                if not callable(predict_profile):
                    raise RuntimeError("OWAFSBNNOptimizer requires a BNN acquisition backend with predict_response_profile")
                if not callable(simulate_afs):
                    raise RuntimeError("OWAFSBNNOptimizer requires a simulator with simulate_afs")
                prediction = predict_profile(selection.requested_geometry)
                simulation = simulate_afs(selection.requested_geometry, prediction)
            t_sim = float(time.perf_counter() - t_sim_start)
            self._sim_elapsed_total += t_sim
            records.append(SimulationRecord(selection=selection, simulation=simulation, t_simulator_sec=t_sim))
        return records

    def collect_post_simulation_data(
        self,
        records: list[SimulationRecord],
        *,
        iteration: int,
    ) -> list[dict[str, Any]]:
        trials = super().collect_post_simulation_data(records, iteration=iteration)
        if iteration < self.n_init:
            return trials

        out: list[dict[str, Any]] = []
        for trial in trials:
            out.append(trial)
            if not self._should_full_validate(trial):
                trial["full_validation_triggered"] = False
                continue
            validation_case_id = self._validation_case_id(trial)
            trial["full_validation_triggered"] = True
            trial["full_validation_case_id"] = validation_case_id
            trial["afs_vf_objective_estimate"] = float(trial["objective"])
            trial["exclude_from_model_training"] = True
            trial["exclude_from_state"] = True
            trial["best_so_far"] = self.state.best_objective
            metadata = dict(trial.get("simulation_metadata") or {})
            metadata.update(
                {
                    "full_validation_triggered": True,
                    "full_validation_case_id": validation_case_id,
                    "afs_vf_objective_estimate": float(trial["objective"]),
                }
            )
            trial["simulation_metadata"] = metadata

            validation_record = self._run_full_validation(trial, validation_case_id, iteration=iteration)
            validation_trial = self._trial_from_record(validation_record, iteration=iteration)
            validation_trial.update(
                {
                    "phase": "validation",
                    "counts_toward_budget": False,
                    "validation_for_case_id": trial.get("case_id"),
                    "validation_for_iteration": int(trial.get("iteration", iteration)),
                    "validation_for_objective_estimate": float(trial["objective"]),
                    "full_validation_triggered": False,
                    "exclude_from_model_training": False,
                    "exclude_from_state": False,
                }
            )
            validation_metadata = dict(validation_trial.get("simulation_metadata") or {})
            validation_metadata.update(
                {
                    "validation_for_case_id": trial.get("case_id"),
                    "validation_for_iteration": int(trial.get("iteration", iteration)),
                    "validation_for_objective_estimate": float(trial["objective"]),
                }
            )
            validation_trial["simulation_metadata"] = validation_metadata
            trial["full_validation_status"] = validation_trial.get("simulation_status")
            trial["full_validation_objective"] = validation_trial.get("objective")
            out.append(validation_trial)
        return out

    def _should_full_validate(self, trial: dict[str, Any]) -> bool:
        if trial.get("simulation_status") != "done":
            return False
        objective = float(trial.get("objective", float("inf")))
        if not np.isfinite(objective):
            return False
        if self.state.best_objective is None:
            return True
        return objective < float(self.state.best_objective)

    def _run_full_validation(
        self,
        trial: dict[str, Any],
        validation_case_id: str,
        *,
        iteration: int,
    ) -> SimulationRecord:
        simulate_full_validation = getattr(self.simulator, "simulate_full_validation", None)
        if not callable(simulate_full_validation):
            raise RuntimeError("OWAFSBNNOptimizer requires a simulator with simulate_full_validation")
        selection = SelectionResult(
            requested_geometry=np.asarray(trial["requested_geometry"], dtype=float),
            predicted_mean=trial.get("predicted_mean"),
            predicted_uncertainty=trial.get("predicted_uncertainty"),
            acquisition_score=trial.get("acquisition_score"),
            selection_debug={
                "mode": "full_validation",
                "model_fit_status": "not_applicable",
                "validation_for_case_id": trial.get("case_id"),
                "validation_for_iteration": int(trial.get("iteration", iteration)),
            },
        )
        t_sim_start = time.perf_counter()
        simulation = simulate_full_validation(selection.requested_geometry, case_id=validation_case_id)
        t_sim = float(time.perf_counter() - t_sim_start)
        self._sim_elapsed_total += t_sim
        return SimulationRecord(selection=selection, simulation=simulation, t_simulator_sec=t_sim)

    def _validation_case_id(self, trial: dict[str, Any]) -> str:
        base = str(trial.get("case_id") or trial.get("job_id") or f"{self.run_id}_it{int(trial['iteration']):04d}")
        return f"{base}_full_validation"
