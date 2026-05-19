from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .botorch_acquisition import AcquisitionFitResult
from .optimizer_base import SelectionResult

try:
    from src.bayesian.modules.loss import BKLLoss
    from src.bayesian.modules.models import MultiOutputBNN
except ImportError:  # pragma: no cover - tests often put src/ directly on PYTHONPATH
    from bayesian.modules.loss import BKLLoss
    from bayesian.modules.models import MultiOutputBNN


@dataclass(frozen=True)
class ScalarBNNConfig:
    """Training settings for the scalar-objective BNN used by OptimizerBoTorch."""

    hidden_dims: tuple[int, ...] = (64, 64)
    prior_sigma: float = 0.1
    epochs: int = 400
    update_epochs: int = 10
    warm_start: bool = True
    initial_train_size: int | None = None
    batch_size: int = 32
    lr: float = 1e-3
    bayesian_loss_weight: float = 1e-4
    mc_samples: int = 64
    acquisition: str = "qlogei"
    beta: float = 2.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ScalarBNNConfig":
        payload = dict(values or {})
        if "hidden_dims" in payload:
            payload["hidden_dims"] = tuple(int(v) for v in payload["hidden_dims"])
        return cls(**payload)


class _TargetStandardizer:
    def __init__(self, y: np.ndarray, *, eps: float = 1e-8) -> None:
        import torch

        arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < eps:
            std = 1.0
        self.mean = torch.tensor([mean], dtype=torch.float32)
        self.std = torch.tensor([std], dtype=torch.float32)

    def regularize(self, y):
        self.mean = self.mean.to(y.device)
        self.std = self.std.to(y.device)
        return (y - self.mean) / self.std

    def deregularize(self, y):
        self.mean = self.mean.to(y.device)
        self.std = self.std.to(y.device)
        return y * self.std + self.mean


class ScalarBNNRegressor:
    """Legacy Bayesian MLP wrapper for scalar objective regression.

    Inputs are expected to be normalized geometry coordinates. Outputs are
    standardized internally and converted back to the original objective scale
    for acquisition scoring.
    """

    output_name = "objective"

    def __init__(
        self,
        *,
        input_dim: int,
        config: ScalarBNNConfig | None = None,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.config = config or ScalarBNNConfig()
        self.seed = int(seed)
        self.device_name = device
        self.device = None
        self.model = None
        self.target_standardizer: _TargetStandardizer | None = None
        self._optimizer = None
        self.trained_epochs = 0

    @property
    def is_fitted(self) -> bool:
        return self.model is not None and self.target_standardizer is not None and self.device is not None

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int | None = None) -> None:
        import torch

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self._validate_training_arrays(X_np, y_np)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device(self.device_name) if self.device_name else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model().to(self.device)
        self.target_standardizer = _TargetStandardizer(y_np)
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))
        epoch_count = int(epochs if epochs is not None else self.config.epochs)
        self._train_arrays(X_np, y_np, epochs=epoch_count, generator_seed=self.seed)
        self.trained_epochs = epoch_count

    def update(self, X: np.ndarray, y: np.ndarray, *, epochs: int) -> None:
        """Continue training the current BNN on an enlarged online dataset."""

        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if not self.is_fitted or self._optimizer is None:
            raise RuntimeError("ScalarBNNRegressor must be fitted before update")
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self._validate_training_arrays(X_np, y_np)
        self._train_arrays(X_np, y_np, epochs=int(epochs), generator_seed=self.seed + self.trained_epochs)
        self.trained_epochs += int(epochs)

    def _validate_training_arrays(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n, {self.input_dim})")
        if X.shape[0] == 0 or y.shape != (X.shape[0], 1):
            raise ValueError("y must have shape (n, 1) and n must be non-zero")

    def _train_arrays(self, X: np.ndarray, y: np.ndarray, *, epochs: int, generator_seed: int) -> None:
        import torch

        if self.model is None or self.device is None or self.target_standardizer is None or self._optimizer is None:
            raise RuntimeError("model, device, target standardizer, and optimizer must be initialized")
        if epochs <= 0:
            raise ValueError("epochs must be > 0")

        X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(int(self.config.batch_size), len(dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(generator_seed)),
        )
        mse = torch.nn.MSELoss()
        bkl = BKLLoss(reduction="mean", last_layer_only=False)

        self.model.train()
        self.model.unfreeze()
        for _ in range(int(epochs)):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = self.target_standardizer.regularize(y_batch.to(self.device))
                self._optimizer.zero_grad()
                pred = self.model(X_batch)[self.output_name]
                loss = mse(pred, y_batch) + float(self.config.bayesian_loss_weight) * bkl(self.model)
                loss.backward()
                self._optimizer.step()

    def predict_samples(self, X: Any, *, mc_samples: int | None = None):
        import torch

        if not self.is_fitted:
            raise RuntimeError("ScalarBNNRegressor is not fitted")
        assert self.model is not None and self.device is not None and self.target_standardizer is not None

        if not torch.is_tensor(X):
            X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=self.device)
        else:
            X = X.to(device=self.device, dtype=torch.float32)

        self.model.eval()
        self.model.unfreeze()
        draws = []
        with torch.no_grad():
            for _ in range(int(mc_samples or self.config.mc_samples)):
                pred = self.model(X)[self.output_name]
                draws.append(self.target_standardizer.deregularize(pred).squeeze(-1))
        return torch.stack(draws, dim=0)

    def posterior_mean_variance(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        samples = self.predict_samples(X)
        mean = samples.mean(dim=0).detach().cpu().numpy()
        var = samples.var(dim=0, unbiased=False).detach().cpu().numpy()
        return np.asarray(mean, dtype=float), np.asarray(var, dtype=float)

    def _build_model(self):
        hidden = list(self.config.hidden_dims)
        prior_mus = [0.0] * max(len(hidden), 1)
        prior_sigmas = [float(self.config.prior_sigma)] * max(len(hidden), 1)
        return MultiOutputBNN(
            input_dim=self.input_dim,
            hidden_dims=hidden,
            prior_mus=prior_mus,
            prior_sigmas=prior_sigmas,
            output_channels={self.output_name: (1, None)},
        )


class BNNExpectedImprovementAcquisition:
    """Discrete-candidate acquisition from BNN posterior samples.

    BoTorch maximizes acquisition values. For this minimization problem, qEI
    with q=1 is `E[max(best_objective - Y, 0)]`. qLogEI is the log of that
    Monte Carlo estimate; LCB is represented as `-(mean - beta*std)`.
    """

    def __init__(
        self,
        surrogate: ScalarBNNRegressor,
        *,
        best_objective: float,
        mode: str = "qlogei",
        beta: float = 2.0,
        mc_samples: int | None = None,
    ) -> None:
        if mode not in {"qlogei", "qei", "ei", "lcb"}:
            raise ValueError("mode must be 'qlogei', 'qei', 'ei', or 'lcb'")
        self.surrogate = surrogate
        self.best_objective = float(best_objective)
        self.mode = mode
        self.beta = float(beta)
        self.mc_samples = mc_samples

    def __call__(self, X):
        import torch

        original_shape = X.shape
        if X.ndim == 3:
            if original_shape[-2] != 1:
                raise ValueError("BNN acquisition currently supports q=1")
            X_eval = X.squeeze(-2)
        elif X.ndim == 2:
            X_eval = X
        else:
            raise ValueError("X must have shape (n, d) or (n, 1, d)")

        samples = self.surrogate.predict_samples(X_eval, mc_samples=self.mc_samples)
        improvement = torch.clamp(self.best_objective - samples, min=0.0)
        if self.mode == "qlogei":
            eps = torch.finfo(samples.dtype).tiny
            values = torch.log(torch.clamp(improvement, min=eps).mean(dim=0))
        elif self.mode in {"qei", "ei"}:
            values = improvement.mean(dim=0)
        else:
            mean = samples.mean(dim=0)
            std = samples.std(dim=0, unbiased=False)
            values = -(mean - self.beta * std)
        return values


class BNNExpectedImprovementBackend:
    """Scalar-objective BNN backend for OptimizerBoTorch.

    This backend owns both the BNN surrogate and the custom acquisition. It is
    passed into OptimizerBoTorch as an input dependency.
    """

    name = "bnn_qlogei"

    def __init__(
        self,
        config: dict[str, Any] | ScalarBNNConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.config = config if isinstance(config, ScalarBNNConfig) else ScalarBNNConfig.from_mapping(config)
        self.device_name = device
        self._torch = None
        self._device = None
        self._surrogate: ScalarBNNRegressor | None = None
        self._last_train_count = 0
        self._last_fit_mode = "not_fitted"

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int | None = None) -> AcquisitionFitResult:
        try:
            import torch
        except ImportError as exc:
            self._surrogate = None
            return AcquisitionFitResult(ready=False, error=f"ImportError: {exc}")

        self._torch = torch
        self._device = torch.device(self.device_name) if self.device_name else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            X_np = np.asarray(X, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            if self._should_warm_start(X_np):
                assert self._surrogate is not None
                self._surrogate.update(X_np, y_np, epochs=int(self.config.update_epochs))
                self._last_fit_mode = "warm_start_update"
            else:
                surrogate = ScalarBNNRegressor(
                    input_dim=X_np.shape[1],
                    config=self.config,
                    seed=0 if seed is None else int(seed),
                    device=str(self._device),
                )
                surrogate.fit(X_np, y_np, epochs=self._initial_epoch_count(len(X_np)))
                self._surrogate = surrogate
                self._last_fit_mode = "initial_fit"
            self._last_train_count = int(len(X_np))
            return AcquisitionFitResult(ready=True)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            self._surrogate = None
            self._last_train_count = 0
            self._last_fit_mode = "failed"
            return AcquisitionFitResult(ready=False, error=f"{type(exc).__name__}: {exc}")

    def _should_warm_start(self, X: np.ndarray) -> bool:
        return (
            bool(self.config.warm_start)
            and self._surrogate is not None
            and self._surrogate.is_fitted
            and int(len(X)) > int(self._last_train_count)
        )

    def _initial_epoch_count(self, train_count: int) -> int:
        initial = self.config.initial_train_size
        if initial is None or train_count <= int(initial):
            return int(self.config.epochs)
        extra_samples = int(train_count) - int(initial)
        return int(self.config.epochs) + extra_samples * int(self.config.update_epochs)

    def select(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any],
    ) -> SelectionResult:
        try:
            from botorch.optim import optimize_acqf_discrete
        except ImportError as exc:
            raise RuntimeError(f"ImportError: {exc}") from exc

        if self._torch is None or self._device is None or self._surrogate is None:
            raise RuntimeError("BNN acquisition backend is not fitted")

        Xcand = self._torch.tensor(candidates_normalized, dtype=self._torch.float32, device=self._device)
        best_objective = min(float(row["objective"]) for row in history if np.isfinite(float(row["objective"])))
        acq = BNNExpectedImprovementAcquisition(
            self._surrogate,
            best_objective=best_objective,
            mode=self.config.acquisition,
            beta=self.config.beta,
            mc_samples=self.config.mc_samples,
        )
        selected_t, values = optimize_acqf_discrete(acq_function=acq, q=1, choices=Xcand)
        scores = acq(Xcand.unsqueeze(1)).detach().cpu().numpy().reshape(-1)
        selected_norm = selected_t[0].detach()
        dists = ((Xcand - selected_norm) ** 2).sum(dim=-1)
        pos = int(dists.argmin().item())
        means, variances = self._surrogate.posterior_mean_variance(Xcand[pos].reshape(1, -1))
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[pos], dtype=float),
            predicted_mean=float(means.reshape(-1)[0]),
            predicted_uncertainty=float(np.sqrt(max(float(variances.reshape(-1)[0]), 0.0))),
            acquisition_score=float(scores[pos]),
            selection_debug={
                "mode": f"bnn_{self.config.acquisition}_discrete",
                "model_fit_status": "ok",
                "selected_pos": pos,
                "best_score": float(values.detach().cpu().numpy().reshape(-1)[0]),
                "candidate_pool_size": int(len(candidates_actual)),
                **tr_debug,
                "bnn_runtime": {
                    "device": str(self._device),
                    "dtype": "torch.float32",
                    "x_is_normalized": True,
                    "mc_samples": int(self.config.mc_samples),
                    "epochs": int(self.config.epochs),
                    "update_epochs": int(self.config.update_epochs),
                    "warm_start": bool(self.config.warm_start),
                    "fit_mode": self._last_fit_mode,
                    "train_count": int(self._last_train_count),
                    "trained_epochs": int(self._surrogate.trained_epochs),
                },
            },
        )
