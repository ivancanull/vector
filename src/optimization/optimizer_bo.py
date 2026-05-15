from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .optimizer_base import OptimizerBase, SelectionResult


class OptimizerBoTorch(OptimizerBase):
    """BoTorch single-objective optimizer over a CandidateSpace."""

    optimizer_backend = "botorch"

    def __init__(
        self,
        *,
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
        self._torch = None
        self._device = None
        self._dtype = None
        self._model = None
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
        selection = self._select_next(self.candidate_space.remaining())
        return [selection]

    def _fit_model(self) -> None:
        valid_trials = [
            row
            for row in self._history
            if row.get("simulation_status") == "done" and np.isfinite(float(row.get("objective", float("inf"))))
        ]
        if not valid_trials:
            self._model_ready = False
            self._last_fit_error = None
            return

        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
            import torch
        except ImportError as exc:
            self._model_ready = False
            self._last_fit_error = f"ImportError: {exc}"
            return

        self._torch = torch
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._dtype = torch.double
        torch.manual_seed(0 if self.seed is None else int(self.seed))
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(0 if self.seed is None else int(self.seed))

        X_np = np.asarray([row["geometry_normalized"] for row in valid_trials], dtype=float)
        y_np = np.asarray([self._extract_scalar_objective(row) for row in valid_trials], dtype=float).reshape(-1, 1)

        try:
            X = torch.tensor(X_np, dtype=self._dtype, device=self._device)
            y = torch.tensor(y_np, dtype=self._dtype, device=self._device)
            model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            self._model = model
            self._model_ready = True
            self._last_fit_error = None
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            self._model = None
            self._model_ready = False
            self._last_fit_error = f"{type(exc).__name__}: {exc}"

    def _select_next(self, candidates_actual: np.ndarray) -> SelectionResult:
        candidates_actual = np.asarray(candidates_actual, dtype=float)
        if len(candidates_actual) == 0:
            raise RuntimeError("No candidates left to select")
        candidates_actual, tr_debug = self._apply_trust_region_filter(candidates_actual)

        if not self._model_ready:
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
            from botorch.acquisition.logei import qLogExpectedImprovement
            from botorch.acquisition.objective import GenericMCObjective
            from botorch.optim import optimize_acqf_discrete
            from botorch.sampling.normal import SobolQMCNormalSampler
        except ImportError as exc:
            return self._deterministic_selection(
                candidates_actual,
                {
                    "mode": "deterministic_import_error",
                    "model_fit_status": "failed",
                    "select_error": f"ImportError: {exc}",
                    "fallback_policy": "first_remaining_candidate",
                    **tr_debug,
                },
            )

        try:
            assert self._torch is not None and self._device is not None and self._dtype is not None
            assert self._model is not None
            Xcand_np = self.candidate_space.normalize(candidates_actual)
            Xcand = self._torch.tensor(Xcand_np, dtype=self._dtype, device=self._device)
            objective = GenericMCObjective(lambda Z, X=None: -Z[..., 0])
            best_f = float(-min(float(row["objective"]) for row in self._history if np.isfinite(float(row["objective"]))))
            sampler = SobolQMCNormalSampler(sample_shape=self._torch.Size([128]), seed=0 if self.seed is None else int(self.seed))
            acq = qLogExpectedImprovement(
                model=self._model,
                best_f=best_f,
                sampler=sampler,
                objective=objective,
            )
            selected_t, values = optimize_acqf_discrete(acq_function=acq, q=1, choices=Xcand)
            scores = acq(Xcand.unsqueeze(1)).detach().cpu().numpy().reshape(-1)
            selected_norm = selected_t[0].detach()
            dists = ((Xcand - selected_norm) ** 2).sum(dim=-1)
            pos = int(dists.argmin().item())
            posterior = self._model.posterior(Xcand[pos].reshape(1, -1))
            mean = float(posterior.mean.detach().cpu().numpy().reshape(-1)[0])
            var = float(posterior.variance.detach().cpu().numpy().reshape(-1)[0])
            return SelectionResult(
                requested_geometry=np.asarray(candidates_actual[pos], dtype=float),
                predicted_mean=mean,
                predicted_uncertainty=float(np.sqrt(max(var, 0.0))),
                acquisition_score=float(scores[pos]),
                selection_debug={
                    "mode": "qlogei_discrete",
                    "model_fit_status": "ok",
                    "selected_pos": pos,
                    "best_score": float(values.detach().cpu().numpy().reshape(-1)[0]),
                    "candidate_pool_size": int(len(candidates_actual)),
                    **tr_debug,
                    "kernel_runtime": {
                        "device": str(self._device),
                        "dtype": "torch.double",
                        "x_is_normalized": True,
                    },
                },
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

    def _deterministic_selection(self, candidates_actual: np.ndarray, debug: dict[str, Any]) -> SelectionResult:
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[0], dtype=float),
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
