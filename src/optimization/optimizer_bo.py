from __future__ import annotations

from typing import Any

import numpy as np

from .optimizer_base import OptimizerBase, SelectionResult


class OptimizerBoTorch(OptimizerBase):
    """BoTorch single-objective optimizer over a CandidateSpace."""

    optimizer_backend = "botorch"

    def __init__(self, *, retrain_every: int = 1, **kwargs: Any) -> None:
        if retrain_every <= 0:
            raise ValueError("retrain_every must be > 0")
        super().__init__(**kwargs)
        self.retrain_every = int(retrain_every)
        self._torch = None
        self._device = None
        self._dtype = None
        self._model = None
        self._model_ready = False
        self._last_fit_error: str | None = None

    def plan_next(self, iteration: int) -> list[SelectionResult]:
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

        if not self._model_ready:
            return self._deterministic_selection(
                candidates_actual,
                {
                    "mode": "deterministic_not_ready",
                    "model_fit_status": "failed" if self._last_fit_error else "not_ready",
                    "fit_error": self._last_fit_error,
                    "fallback_policy": "first_remaining_candidate",
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
                },
            )

    def _deterministic_selection(self, candidates_actual: np.ndarray, debug: dict[str, Any]) -> SelectionResult:
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[0], dtype=float),
            selection_debug=debug,
        )

    @staticmethod
    def _extract_scalar_objective(row: dict[str, Any]) -> float:
        arr = np.asarray(row.get("objective_array", []), dtype=float).reshape(-1)
        if arr.shape != (1,):
            raise ValueError(f"Only scalar objective arrays are supported now; got shape {arr.shape}")
        return float(arr[0])


BoTorchOptimizer = OptimizerBoTorch
OnlineBOOptimizer = OptimizerBoTorch
