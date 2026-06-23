from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .optimizer_base import SelectionResult


@dataclass(frozen=True)
class AcquisitionFitResult:
    ready: bool
    error: str | None = None


@dataclass(frozen=True)
class AcquisitionScoreResult:
    predicted_mean: np.ndarray
    predicted_uncertainty: np.ndarray
    acquisition_score: np.ndarray
    debug: dict[str, Any]


class BoTorchAcquisitionBackend(Protocol):
    """Adapter interface used by OptimizerBoTorch.

    Implementations own the surrogate model and acquisition function. The
    optimizer supplies normalized training data plus either finite candidates
    or continuous-domain bounds, depending on the candidate-space type.
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

    def score_candidates(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any] | None = None,
    ) -> AcquisitionScoreResult:
        ...


def _project_normalized_candidate(candidate_space: Any, selected_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_norm = np.asarray(selected_norm, dtype=float).reshape(-1)
    actual = np.asarray(candidate_space.denormalize(selected_norm), dtype=float).reshape(-1)
    project = getattr(candidate_space, "project", None)
    if callable(project):
        actual = np.asarray(project(actual), dtype=float).reshape(-1)
    normalized = np.asarray(candidate_space.normalize(actual), dtype=float).reshape(-1)
    return actual, normalized


def _is_duplicate_actual(
    actual: np.ndarray,
    evaluated_actual: np.ndarray | None,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if evaluated_actual is None:
        return False
    evaluated = np.asarray(evaluated_actual, dtype=float)
    if evaluated.size == 0:
        return False
    evaluated = evaluated.reshape(-1, len(np.asarray(actual).reshape(-1)))
    target = np.asarray(actual, dtype=float).reshape(1, -1)
    return bool(np.any(np.all(np.isclose(evaluated, target, atol=atol, rtol=rtol), axis=1)))


def _continuous_gp_selection(
    *,
    model: Any,
    torch: Any,
    device: Any,
    dtype: Any,
    seed: int,
    history: list[dict[str, Any]],
    bounds_normalized: np.ndarray,
    candidate_space: Any,
    tr_debug: dict[str, Any],
    evaluated_actual: np.ndarray | None,
    num_restarts: int,
    raw_samples: int,
    maxiter: int,
    fallback_samples: int,
    acquisition: str,
) -> SelectionResult:
    if acquisition == "qlogei":
        from botorch.acquisition.logei import qLogExpectedImprovement

        acquisition_cls = qLogExpectedImprovement
        mode = "gp_qlogei_continuous"
    elif acquisition == "qei":
        from botorch.acquisition.monte_carlo import qExpectedImprovement

        acquisition_cls = qExpectedImprovement
        mode = "gp_qei_continuous"
    else:
        raise ValueError(f"Unsupported continuous GP acquisition={acquisition!r}")

    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler

    bounds_np = np.asarray(bounds_normalized, dtype=float)
    if bounds_np.ndim != 2 or bounds_np.shape[0] != 2:
        raise ValueError("bounds_normalized must have shape (2, d)")
    if np.any(bounds_np[1] <= bounds_np[0]):
        raise ValueError("bounds_normalized upper bounds must be greater than lower bounds")

    torch.manual_seed(int(seed))
    if getattr(device, "type", "") == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    objective = GenericMCObjective(lambda Z, X=None: -Z[..., 0])
    best_f = float(-min(float(row["objective"]) for row in history if np.isfinite(float(row["objective"]))))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=int(seed))
    acq = acquisition_cls(
        model=model,
        best_f=best_f,
        sampler=sampler,
        objective=objective,
    )
    bounds_t = torch.tensor(bounds_np, dtype=dtype, device=device)
    selected_t, values = optimize_acqf(
        acq_function=acq,
        bounds=bounds_t,
        q=1,
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        options={"maxiter": int(maxiter)},
    )
    selected_norm = selected_t.detach().cpu().numpy().reshape(-1)
    actual, selected_norm = _project_normalized_candidate(candidate_space, selected_norm)

    atol = float(getattr(candidate_space, "match_atol", 1e-9))
    rtol = float(getattr(candidate_space, "match_rtol", 1e-9))
    duplicate_after_projection = _is_duplicate_actual(actual, evaluated_actual, atol=atol, rtol=rtol)
    fallback_used = False
    fallback_candidate_count = 0
    if duplicate_after_projection and int(fallback_samples) > 0:
        rng = np.random.default_rng(int(seed) + 1009 * max(1, len(history)))
        samples_norm = rng.uniform(bounds_np[0], bounds_np[1], size=(int(fallback_samples), bounds_np.shape[1]))
        sample_actual = np.asarray(candidate_space.denormalize(samples_norm), dtype=float)
        project = getattr(candidate_space, "project", None)
        if callable(project):
            sample_actual = np.asarray(project(sample_actual), dtype=float)
        sample_norm = np.asarray(candidate_space.normalize(sample_actual), dtype=float)
        keep = [
            not _is_duplicate_actual(row, evaluated_actual, atol=atol, rtol=rtol)
            for row in np.asarray(sample_actual, dtype=float)
        ]
        if any(keep):
            sample_actual = np.asarray(sample_actual[keep], dtype=float)
            sample_norm = np.asarray(sample_norm[keep], dtype=float)
            fallback_candidate_count = int(len(sample_actual))
            Xfallback = torch.tensor(sample_norm, dtype=dtype, device=device)
            fallback_scores = acq(Xfallback.unsqueeze(1)).detach().cpu().numpy().reshape(-1)
            pos = int(np.argmax(fallback_scores))
            actual = np.asarray(sample_actual[pos], dtype=float)
            selected_norm = np.asarray(sample_norm[pos], dtype=float)
            fallback_used = True

    Xsel = torch.tensor(selected_norm.reshape(1, -1), dtype=dtype, device=device)
    score = float(acq(Xsel.unsqueeze(1)).detach().cpu().numpy().reshape(-1)[0])
    posterior = model.posterior(Xsel)
    mean = float(posterior.mean.detach().cpu().numpy().reshape(-1)[0])
    var = float(posterior.variance.detach().cpu().numpy().reshape(-1)[0])
    return SelectionResult(
        requested_geometry=np.asarray(actual, dtype=float),
        predicted_mean=mean,
        predicted_uncertainty=float(np.sqrt(max(var, 0.0))),
        acquisition_score=score,
        selection_debug={
            "mode": mode,
            "model_fit_status": "ok",
            "best_score": score,
            "raw_optimized_score": float(values.detach().cpu().numpy().reshape(-1)[0]),
            "bounds_normalized": bounds_np.tolist(),
            "selected_normalized": selected_norm.tolist(),
            "duplicate_after_projection": bool(duplicate_after_projection),
            "fallback_used": bool(fallback_used),
            "fallback_candidate_count": int(fallback_candidate_count),
            **tr_debug,
            "kernel_runtime": {
                "device": str(device),
                "dtype": "torch.double",
                "x_is_normalized": True,
                "acquisition": acquisition,
                "optimizer": "optimize_acqf",
                "num_restarts": int(num_restarts),
                "raw_samples": int(raw_samples),
                "maxiter": int(maxiter),
            },
        },
    )


class GPqLogEIAcquisitionBackend:
    """Default GP + logEI backend used by OptimizerBoTorch."""

    name = "gp_qlogei"

    def __init__(
        self,
        *,
        num_restarts: int = 20,
        raw_samples: int = 512,
        maxiter: int = 200,
        fallback_samples: int = 2048,
    ) -> None:
        if num_restarts <= 0:
            raise ValueError("num_restarts must be > 0")
        if raw_samples <= 0:
            raise ValueError("raw_samples must be > 0")
        if maxiter <= 0:
            raise ValueError("maxiter must be > 0")
        if fallback_samples < 0:
            raise ValueError("fallback_samples must be >= 0")
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.maxiter = int(maxiter)
        self.fallback_samples = int(fallback_samples)
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
        scores = self.score_candidates(
            candidates_actual=candidates_actual,
            candidates_normalized=candidates_normalized,
            history=history,
            tr_debug=tr_debug,
        )
        pos = int(np.argmax(scores.acquisition_score))
        debug = dict(scores.debug)
        debug.update({"selected_pos": pos, "best_score": float(scores.acquisition_score[pos])})
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[pos], dtype=float),
            predicted_mean=float(scores.predicted_mean[pos]),
            predicted_uncertainty=float(scores.predicted_uncertainty[pos]),
            acquisition_score=float(scores.acquisition_score[pos]),
            selection_debug=debug,
        )

    def score_candidates(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any] | None = None,
    ) -> AcquisitionScoreResult:
        try:
            from botorch.acquisition.logei import qLogExpectedImprovement
            from botorch.acquisition.objective import GenericMCObjective
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
        scores = acq(Xcand.unsqueeze(1)).detach().cpu().numpy().reshape(-1)
        posterior = self._model.posterior(Xcand)
        means = posterior.mean.detach().cpu().numpy().reshape(-1)
        variances = posterior.variance.detach().cpu().numpy().reshape(-1)
        uncertainties = np.sqrt(np.maximum(variances, 0.0))
        return AcquisitionScoreResult(
            predicted_mean=np.asarray(means, dtype=float),
            predicted_uncertainty=np.asarray(uncertainties, dtype=float),
            acquisition_score=np.asarray(scores, dtype=float),
            debug={
                "mode": "gp_qlogei_discrete",
                "model_fit_status": "ok",
                "candidate_pool_size": int(len(candidates_actual)),
                **(tr_debug or {}),
                "kernel_runtime": {
                    "device": str(self._device),
                    "dtype": "torch.double",
                    "x_is_normalized": True,
                    "acquisition": "qlogei",
                },
            },
        )

    def select_continuous(
        self,
        *,
        bounds_normalized: np.ndarray,
        candidate_space: Any,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any],
        evaluated_actual: np.ndarray | None = None,
    ) -> SelectionResult:
        if self._torch is None or self._device is None or self._dtype is None or self._model is None:
            raise RuntimeError("GP acquisition backend is not fitted")
        return _continuous_gp_selection(
            model=self._model,
            torch=self._torch,
            device=self._device,
            dtype=self._dtype,
            seed=self._seed,
            history=history,
            bounds_normalized=bounds_normalized,
            candidate_space=candidate_space,
            tr_debug=tr_debug,
            evaluated_actual=evaluated_actual,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            maxiter=self.maxiter,
            fallback_samples=self.fallback_samples,
            acquisition="qlogei",
        )


class GPExpectedImprovementBackend:
    """GP backend using BoTorch qExpectedImprovement for qEI comparisons."""

    name = "gp_qei"

    def __init__(
        self,
        *,
        num_restarts: int = 20,
        raw_samples: int = 512,
        maxiter: int = 200,
        fallback_samples: int = 2048,
    ) -> None:
        if num_restarts <= 0:
            raise ValueError("num_restarts must be > 0")
        if raw_samples <= 0:
            raise ValueError("raw_samples must be > 0")
        if maxiter <= 0:
            raise ValueError("maxiter must be > 0")
        if fallback_samples < 0:
            raise ValueError("fallback_samples must be >= 0")
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.maxiter = int(maxiter)
        self.fallback_samples = int(fallback_samples)
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
        scores = self.score_candidates(
            candidates_actual=candidates_actual,
            candidates_normalized=candidates_normalized,
            history=history,
            tr_debug=tr_debug,
        )
        pos = int(np.argmax(scores.acquisition_score))
        debug = dict(scores.debug)
        debug.update({"selected_pos": pos, "best_score": float(scores.acquisition_score[pos])})
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[pos], dtype=float),
            predicted_mean=float(scores.predicted_mean[pos]),
            predicted_uncertainty=float(scores.predicted_uncertainty[pos]),
            acquisition_score=float(scores.acquisition_score[pos]),
            selection_debug=debug,
        )

    def score_candidates(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any] | None = None,
    ) -> AcquisitionScoreResult:
        try:
            from botorch.acquisition.monte_carlo import qExpectedImprovement
            from botorch.acquisition.objective import GenericMCObjective
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
        scores = acq(Xcand.unsqueeze(1)).detach().cpu().numpy().reshape(-1)
        posterior = self._model.posterior(Xcand)
        means = posterior.mean.detach().cpu().numpy().reshape(-1)
        variances = posterior.variance.detach().cpu().numpy().reshape(-1)
        uncertainties = np.sqrt(np.maximum(variances, 0.0))
        return AcquisitionScoreResult(
            predicted_mean=np.asarray(means, dtype=float),
            predicted_uncertainty=np.asarray(uncertainties, dtype=float),
            acquisition_score=np.asarray(scores, dtype=float),
            debug={
                "mode": "gp_qei_discrete",
                "model_fit_status": "ok",
                "candidate_pool_size": int(len(candidates_actual)),
                **(tr_debug or {}),
                "kernel_runtime": {
                    "device": str(self._device),
                    "dtype": "torch.double",
                    "x_is_normalized": True,
                    "acquisition": "qei",
                },
            },
        )

    def select_continuous(
        self,
        *,
        bounds_normalized: np.ndarray,
        candidate_space: Any,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any],
        evaluated_actual: np.ndarray | None = None,
    ) -> SelectionResult:
        if self._torch is None or self._device is None or self._dtype is None or self._model is None:
            raise RuntimeError("GP qEI acquisition backend is not fitted")
        return _continuous_gp_selection(
            model=self._model,
            torch=self._torch,
            device=self._device,
            dtype=self._dtype,
            seed=self._seed,
            history=history,
            bounds_normalized=bounds_normalized,
            candidate_space=candidate_space,
            tr_debug=tr_debug,
            evaluated_actual=evaluated_actual,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            maxiter=self.maxiter,
            fallback_samples=self.fallback_samples,
            acquisition="qei",
        )


GPqExpectedImprovementBackend = GPExpectedImprovementBackend
