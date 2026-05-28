from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .optimizer_base import SelectionResult


@dataclass(frozen=True)
class AcquisitionFitResult:
    ready: bool
    error: str | None = None


class BoTorchAcquisitionBackend(Protocol):
    """Adapter interface used by OptimizerBoTorch.

    Implementations own the surrogate model and acquisition function. The
    optimizer only supplies normalized training data and remaining candidates.
    """

    name: str

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> AcquisitionFitResult:
        ...

    def select(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any],
    ) -> SelectionResult:
        ...


class GPqLogEIAcquisitionBackend:
    """Default GP + qLogEI backend used by OptimizerBoTorch."""

    name = "gp_qlogei"

    def __init__(self) -> None:
        self._torch = None
        self._device = None
        self._dtype = None
        self._model = None
        self._seed = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> AcquisitionFitResult:
        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
            import torch
        except ImportError as exc:
            self._model = None
            return AcquisitionFitResult(ready=False, error=f"ImportError: {exc}")

        self._torch = torch
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._dtype = torch.double
        self._seed = 0 if seed is None else int(seed)
        torch.manual_seed(self._seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)

        try:
            X_t = torch.tensor(np.asarray(X, dtype=float), dtype=self._dtype, device=self._device)
            y_t = torch.tensor(np.asarray(y, dtype=float).reshape(-1, 1), dtype=self._dtype, device=self._device)
            model = SingleTaskGP(X_t, y_t, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            self._model = model
            return AcquisitionFitResult(ready=True)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            self._model = None
            return AcquisitionFitResult(ready=False, error=f"{type(exc).__name__}: {exc}")

    def select(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any],
    ) -> SelectionResult:
        try:
            from botorch.acquisition.logei import qLogExpectedImprovement
            from botorch.acquisition.objective import GenericMCObjective
            from botorch.optim import optimize_acqf_discrete
            from botorch.sampling.normal import SobolQMCNormalSampler
        except ImportError as exc:
            raise RuntimeError(f"ImportError: {exc}") from exc

        if self._torch is None or self._device is None or self._dtype is None or self._model is None:
            raise RuntimeError("GP acquisition backend is not fitted")

        Xcand = self._torch.tensor(candidates_normalized, dtype=self._dtype, device=self._device)
        objective = GenericMCObjective(lambda Z, X=None: -Z[..., 0])
        best_f = float(-min(float(row["objective"]) for row in history if np.isfinite(float(row["objective"]))))
        sampler = SobolQMCNormalSampler(sample_shape=self._torch.Size([128]), seed=self._seed)
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
                "mode": "gp_qlogei_discrete",
                "model_fit_status": "ok",
                "selected_pos": pos,
                "best_score": float(values.detach().cpu().numpy().reshape(-1)[0]),
                "candidate_pool_size": int(len(candidates_actual)),
                **tr_debug,
                "kernel_runtime": {
                    "device": str(self._device),
                    "dtype": "torch.double",
                    "x_is_normalized": True,
                    "acquisition": "qlogei",
                },
            },
        )


class GPExpectedImprovementBackend:
    """GP backend using BoTorch qExpectedImprovement for qEI comparisons."""

    name = "gp_qei"

    def __init__(self) -> None:
        self._torch = None
        self._device = None
        self._dtype = None
        self._model = None
        self._seed = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> AcquisitionFitResult:
        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
            import torch
        except ImportError as exc:
            self._model = None
            return AcquisitionFitResult(ready=False, error=f"ImportError: {exc}")

        self._torch = torch
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._dtype = torch.double
        self._seed = 0 if seed is None else int(seed)
        torch.manual_seed(self._seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)

        try:
            X_t = torch.tensor(np.asarray(X, dtype=float), dtype=self._dtype, device=self._device)
            y_t = torch.tensor(np.asarray(y, dtype=float).reshape(-1, 1), dtype=self._dtype, device=self._device)
            model = SingleTaskGP(X_t, y_t, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            self._model = model
            return AcquisitionFitResult(ready=True)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            self._model = None
            return AcquisitionFitResult(ready=False, error=f"{type(exc).__name__}: {exc}")

    def select(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any],
    ) -> SelectionResult:
        try:
            from botorch.acquisition.monte_carlo import qExpectedImprovement
            from botorch.acquisition.objective import GenericMCObjective
            from botorch.optim import optimize_acqf_discrete
            from botorch.sampling.normal import SobolQMCNormalSampler
        except ImportError as exc:
            raise RuntimeError(f"ImportError: {exc}") from exc

        if self._torch is None or self._device is None or self._dtype is None or self._model is None:
            raise RuntimeError("GP qEI acquisition backend is not fitted")

        Xcand = self._torch.tensor(candidates_normalized, dtype=self._dtype, device=self._device)
        objective = GenericMCObjective(lambda Z, X=None: -Z[..., 0])
        best_f = float(-min(float(row["objective"]) for row in history if np.isfinite(float(row["objective"]))))
        sampler = SobolQMCNormalSampler(sample_shape=self._torch.Size([128]), seed=self._seed)
        acq = qExpectedImprovement(
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
                "mode": "gp_qei_discrete",
                "model_fit_status": "ok",
                "selected_pos": pos,
                "best_score": float(values.detach().cpu().numpy().reshape(-1)[0]),
                "candidate_pool_size": int(len(candidates_actual)),
                **tr_debug,
                "kernel_runtime": {
                    "device": str(self._device),
                    "dtype": "torch.double",
                    "x_is_normalized": True,
                    "acquisition": "qei",
                },
            },
        )


GPqExpectedImprovementBackend = GPExpectedImprovementBackend
