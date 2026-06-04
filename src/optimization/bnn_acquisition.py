from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .botorch_acquisition import AcquisitionFitResult
from .optimizer_base import SelectionResult
from .optimization_spec import OptimizationSpec

try:
    from src.modeling import BCLFResponseDataset, FourierBNNResponseSurrogate, LegacyBNNResponseSurrogate
except ImportError:  # pragma: no cover - tests often put src/ directly on PYTHONPATH
    from modeling import BCLFResponseDataset, FourierBNNResponseSurrogate, LegacyBNNResponseSurrogate

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
    checkpoint_path: str | None = None
    force_retrain: bool = False
    require_checkpoint: bool = False
    save_checkpoint: bool = True
    surrogate_variant: str = "raw"
    fourier_bands: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    include_raw_frequency: bool = True
    frequency_min_hz: float | None = None
    frequency_max_hz: float | None = None

    def __post_init__(self) -> None:
        variant = str(self.surrogate_variant).strip().lower()
        if variant not in {"raw", "fourier"}:
            raise ValueError("surrogate_variant must be 'raw' or 'fourier'")
        bands = tuple(float(v) for v in self.fourier_bands)
        if variant == "fourier" and not bands:
            raise ValueError("fourier_bands must be non-empty when surrogate_variant='fourier'")
        object.__setattr__(self, "surrogate_variant", variant)
        object.__setattr__(self, "fourier_bands", bands)

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ScalarBNNConfig":
        payload = dict(values or {})
        if "hidden_dims" in payload:
            payload["hidden_dims"] = tuple(int(v) for v in payload["hidden_dims"])
        if "fourier_bands" in payload and payload["fourier_bands"] is not None:
            payload["fourier_bands"] = tuple(float(v) for v in payload["fourier_bands"])
        if payload.get("checkpoint_path") is not None:
            payload["checkpoint_path"] = str(payload["checkpoint_path"])
        return cls(**payload)


class BNNCheckpointError(Exception):
    """Raised when an explicitly configured BNN checkpoint cannot be used."""


@dataclass(frozen=True)
class BNNResponsePrediction:
    freqs: np.ndarray
    mean_s_params: np.ndarray
    response_samples: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "freqs", np.asarray(self.freqs, dtype=float).reshape(-1))
        object.__setattr__(self, "mean_s_params", np.asarray(self.mean_s_params, dtype=complex))
        object.__setattr__(self, "response_samples", np.asarray(self.response_samples, dtype=complex))


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

    BoTorch maximizes acquisition values. For this minimization problem, EI
    is `E[max(best_objective - Y, 0)]`. The historical `qlogei` mode computes
    logEI from that Monte Carlo estimate; LCB is represented as
    `-(mean - beta*std)`.
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
    """Response-surrogate BNN backend for OptimizerBoTorch.

    The BNN learns S-parameter response curves. Acquisition scores are computed
    by converting BNN response samples back through the optimizer's
    OptimizationSpec, so the EI target is always the same metric used to score
    real simulations.
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
        self._surrogate: LegacyBNNResponseSurrogate | None = None
        self._train_dataset: BCLFResponseDataset | None = None
        self._optimization_spec: OptimizationSpec | None = None
        self._last_train_count = 0
        self._last_fit_mode = "not_fitted"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> AcquisitionFitResult:
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
            _ = np.asarray(X, dtype=np.float32)
            _ = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            dataset = self._response_dataset_from_fit_kwargs(kwargs)
            optimization_spec = kwargs.get("optimization_spec")
            if optimization_spec is None:
                raise ValueError("BNN response acquisition requires optimization_spec")
            self._optimization_spec = optimization_spec
            train_indices = np.arange(dataset.n_samples, dtype=int)
            checkpoint_changed = False
            if self._should_warm_start(dataset):
                assert self._surrogate is not None
                self._surrogate.update(dataset, train_indices, epochs=int(self.config.update_epochs))
                self._last_fit_mode = "warm_start_update"
                checkpoint_changed = True
            elif (
                self._surrogate is not None
                and self._surrogate.is_fitted
                and int(dataset.n_samples) == int(self._last_train_count)
            ):
                self._last_fit_mode = "reuse_fitted"
            else:
                loaded_train_count = self._load_checkpoint_if_available(dataset)
                if loaded_train_count is not None:
                    assert self._surrogate is not None
                    stale_count = int(dataset.n_samples) - int(loaded_train_count)
                    if stale_count > 0:
                        # A persisted checkpoint can lag the optimizer history after an interrupted run.
                        # Continue from the checkpoint for the missed online updates instead of refitting.
                        self._surrogate.update(
                            dataset,
                            train_indices,
                            epochs=int(stale_count) * int(self.config.update_epochs),
                        )
                        self._last_fit_mode = "checkpoint_load_warm_start_update"
                        checkpoint_changed = True
                    else:
                        self._last_fit_mode = "checkpoint_load"
                else:
                    surrogate = self._build_response_surrogate(dataset, seed=seed)
                    surrogate.fit(dataset, train_indices, epochs=self._initial_epoch_count(dataset.n_samples))
                    self._surrogate = surrogate
                    self._last_fit_mode = "initial_fit"
                    checkpoint_changed = True
            self._train_dataset = dataset
            self._last_train_count = int(dataset.n_samples)
            if checkpoint_changed:
                self._save_checkpoint_if_configured(dataset, seed=seed)
            return AcquisitionFitResult(ready=True)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            self._surrogate = None
            self._train_dataset = None
            self._optimization_spec = None
            self._last_train_count = 0
            self._last_fit_mode = "failed"
            return AcquisitionFitResult(ready=False, error=f"{type(exc).__name__}: {exc}")

    def _checkpoint_path(self) -> Path | None:
        return None if self.config.checkpoint_path is None else Path(self.config.checkpoint_path)

    def _load_checkpoint_if_available(self, dataset: BCLFResponseDataset) -> int | None:
        path = self._checkpoint_path()
        if path is None or bool(self.config.force_retrain):
            return None
        if not path.exists():
            if bool(self.config.require_checkpoint):
                raise BNNCheckpointError(
                    f"BNN checkpoint is required but does not exist: {path}. "
                    "Set force_retrain=True to train and write a new checkpoint."
                )
            return None
        try:
            surrogate = self._surrogate_class().load(path, device=str(self._device))
        except Exception as exc:  # noqa: BLE001 - explicit checkpoints should fail loudly.
            raise BNNCheckpointError(f"Failed to load BNN checkpoint {path}: {type(exc).__name__}: {exc}") from exc
        self._validate_checkpoint_surrogate(surrogate, dataset, path)
        self._surrogate = surrogate
        metadata = dict(getattr(surrogate, "checkpoint_metadata", {}) or {})
        train_count = metadata.get("train_count")
        if train_count is None:
            return int(dataset.n_samples)
        train_count = int(train_count)
        if train_count > int(dataset.n_samples):
            raise BNNCheckpointError(
                f"BNN checkpoint {path} was trained with {train_count} samples, "
                f"but the current optimizer history has only {int(dataset.n_samples)}."
            )
        return train_count

    def _surrogate_class(self) -> type[LegacyBNNResponseSurrogate]:
        if self.config.surrogate_variant == "fourier":
            return FourierBNNResponseSurrogate
        return LegacyBNNResponseSurrogate

    def _build_response_surrogate(
        self,
        dataset: BCLFResponseDataset,
        *,
        seed: int | None,
    ) -> LegacyBNNResponseSurrogate:
        kwargs: dict[str, Any] = {
            "hidden_dims": list(self.config.hidden_dims),
            "prior_sigma": float(self.config.prior_sigma),
            "epochs": int(self._initial_epoch_count(dataset.n_samples)),
            "batch_size": int(self.config.batch_size),
            "lr": float(self.config.lr),
            "bayesian_loss_weight": float(self.config.bayesian_loss_weight),
            "mc_samples": int(self.config.mc_samples),
            "target_mode": "db",
            "seed": 0 if seed is None else int(seed),
            "device": str(self._device),
        }
        if self.config.surrogate_variant == "fourier":
            kwargs.update(
                {
                    "fourier_bands": list(self.config.fourier_bands),
                    "include_raw_frequency": bool(self.config.include_raw_frequency),
                    "frequency_min_hz": float(
                        np.min(dataset.freqs)
                        if self.config.frequency_min_hz is None
                        else self.config.frequency_min_hz
                    ),
                    "frequency_max_hz": float(
                        np.max(dataset.freqs)
                        if self.config.frequency_max_hz is None
                        else self.config.frequency_max_hz
                    ),
                }
            )
        return self._surrogate_class()(**kwargs)

    def _expected_surrogate_input_dim(self, dataset: BCLFResponseDataset) -> int:
        geometry_dim = int(dataset.X.shape[1])
        if self.config.surrogate_variant == "fourier":
            raw_dim = 1 if self.config.include_raw_frequency else 0
            return geometry_dim + raw_dim + 2 * len(self.config.fourier_bands)
        return geometry_dim + 1

    def _validate_checkpoint_surrogate(
        self,
        surrogate: LegacyBNNResponseSurrogate,
        dataset: BCLFResponseDataset,
        path: Path,
    ) -> None:
        expected_variant = str(self.config.surrogate_variant)
        actual_variant = str(getattr(surrogate, "surrogate_variant", "raw"))
        if actual_variant != expected_variant:
            raise BNNCheckpointError(
                f"BNN checkpoint {path} surrogate_variant={actual_variant!r} does not match "
                f"configured {expected_variant!r}."
            )
        expected_input_dim = self._expected_surrogate_input_dim(dataset)
        if int(surrogate.input_dim) != expected_input_dim:
            raise BNNCheckpointError(
                f"BNN checkpoint {path} input_dim={surrogate.input_dim} does not match "
                f"expected input_dim={expected_input_dim}."
            )
        if surrogate.target_mode != "db":
            raise BNNCheckpointError(
                f"BNN checkpoint {path} target_mode={surrogate.target_mode!r} does not match required 'db'."
            )
        metadata = dict(getattr(surrogate, "checkpoint_metadata", {}) or {})
        variable_order = metadata.get("variable_order")
        if variable_order is not None and list(variable_order) != list(dataset.variable_order):
            raise BNNCheckpointError(f"BNN checkpoint {path} variable_order does not match current dataset.")
        n_freqs = metadata.get("n_freqs")
        if n_freqs is not None and int(n_freqs) != int(dataset.n_freqs):
            raise BNNCheckpointError(f"BNN checkpoint {path} n_freqs={n_freqs} does not match {dataset.n_freqs}.")

    def _save_checkpoint_if_configured(self, dataset: BCLFResponseDataset, *, seed: int | None) -> None:
        path = self._checkpoint_path()
        if path is None or not bool(self.config.save_checkpoint):
            return
        if self._surrogate is None:
            raise RuntimeError("Cannot save BNN checkpoint before fitting a surrogate")
        self._surrogate.save(
            path,
            metadata={
                "backend": self.name,
                "train_count": int(dataset.n_samples),
                "variable_order": list(dataset.variable_order),
                "n_freqs": int(dataset.n_freqs),
                "freq_min_hz": float(np.min(dataset.freqs)),
                "freq_max_hz": float(np.max(dataset.freqs)),
                "seed": None if seed is None else int(seed),
                "acquisition": str(self.config.acquisition),
                "initial_train_size": self.config.initial_train_size,
                "update_epochs": int(self.config.update_epochs),
            },
        )

    def _should_warm_start(self, dataset: BCLFResponseDataset) -> bool:
        return (
            bool(self.config.warm_start)
            and self._surrogate is not None
            and self._surrogate.is_fitted
            and int(dataset.n_samples) > int(self._last_train_count)
        )

    def _initial_epoch_count(self, train_count: int) -> int:
        initial = self.config.initial_train_size
        if initial is None or train_count <= int(initial):
            return int(self.config.epochs)
        extra_samples = int(train_count) - int(initial)
        return int(self.config.epochs) + extra_samples * int(self.config.update_epochs)

    def _response_dataset_from_fit_kwargs(self, kwargs: dict[str, Any]) -> BCLFResponseDataset:
        history = list(kwargs.get("history") or [])
        X_actual = np.asarray(kwargs.get("X_actual"), dtype=float)
        variable_order = list(kwargs.get("variable_order") or [])
        if X_actual.ndim != 2 or X_actual.shape[0] == 0:
            raise ValueError("BNN response acquisition requires non-empty X_actual")
        if not variable_order:
            raise ValueError("BNN response acquisition requires variable_order")
        if X_actual.shape[1] != len(variable_order):
            raise ValueError("X_actual dimension must match variable_order")
        if len(history) != X_actual.shape[0]:
            raise ValueError("history length must match X_actual rows")

        freqs_ref: np.ndarray | None = None
        s_rows: list[np.ndarray] = []
        case_ids: list[str] = []
        for pos, row in enumerate(history):
            freqs = row.get("_response_freqs")
            s_params = row.get("_response_s_params")
            if freqs is None or s_params is None:
                raise ValueError("history rows are missing in-memory response arrays")
            freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
            s_arr = np.asarray(s_params, dtype=complex)
            if freqs_ref is None:
                freqs_ref = freqs_arr
            elif freqs_arr.shape != freqs_ref.shape or not np.allclose(freqs_arr, freqs_ref):
                raise ValueError("response frequency grids must match for BNN response acquisition")
            if s_arr.ndim != 3 or s_arr.shape[-1] != len(freqs_arr):
                raise ValueError("s_params must have shape (ports, ports, n_freq)")
            s_rows.append(s_arr)
            case_ids.append(str(row.get("case_id") or f"train_{pos}"))

        if freqs_ref is None:
            raise ValueError("no response rows available for BNN response acquisition")
        return BCLFResponseDataset(
            variable_order=variable_order,
            freqs=freqs_ref,
            X=X_actual,
            s_params=np.asarray(s_rows, dtype=complex),
            filenames=[f"{case_id}.s2p" for case_id in case_ids],
            keys=[str(i) for i in range(len(s_rows))],
            case_ids=case_ids,
        )

    def _candidate_dataset(self, candidates_actual: np.ndarray) -> BCLFResponseDataset:
        if self._train_dataset is None:
            raise RuntimeError("BNN response acquisition backend is not fitted")
        X = np.asarray(candidates_actual, dtype=float)
        if X.ndim != 2:
            raise ValueError("candidates_actual must be a 2D array")
        dummy = np.zeros((X.shape[0], 2, 2, self._train_dataset.n_freqs), dtype=complex)
        return BCLFResponseDataset(
            variable_order=list(self._train_dataset.variable_order),
            freqs=self._train_dataset.freqs,
            X=X,
            s_params=dummy,
            filenames=[f"candidate_{i}.s2p" for i in range(X.shape[0])],
            keys=[str(i) for i in range(X.shape[0])],
            case_ids=[f"candidate_{i}" for i in range(X.shape[0])],
        )

    def predict_response_profile(
        self,
        geometry: np.ndarray,
        *,
        mc_samples: int | None = None,
    ) -> BNNResponsePrediction:
        if self._surrogate is None or self._train_dataset is None:
            raise RuntimeError("BNN acquisition backend is not fitted")
        candidate_dataset = self._candidate_dataset(np.asarray(geometry, dtype=float).reshape(1, -1))
        mean, samples = self._surrogate.predict_s_params(
            candidate_dataset,
            np.asarray([0], dtype=int),
            mc_samples=int(mc_samples or self.config.mc_samples),
            return_samples=True,
        )
        return BNNResponsePrediction(
            freqs=np.asarray(self._train_dataset.freqs, dtype=float),
            mean_s_params=np.asarray(mean[0], dtype=complex),
            response_samples=np.asarray(samples[:, 0, :, :, :], dtype=complex),
        )

    def _metric_samples_from_response_samples(self, response_samples: np.ndarray) -> np.ndarray:
        if self._optimization_spec is None or self._train_dataset is None:
            raise RuntimeError("BNN response acquisition backend is not fitted")
        samples = np.asarray(response_samples, dtype=complex)
        if samples.ndim != 5:
            raise ValueError("response_samples must have shape (mc, n, ports, ports, n_freq)")
        metric_rows: list[list[float]] = []
        for sample_idx in range(samples.shape[0]):
            row: list[float] = []
            for candidate_idx in range(samples.shape[1]):
                value = self._optimization_spec.evaluate(
                    self._train_dataset.freqs,
                    samples[sample_idx, candidate_idx],
                ).scalar_objective()
                row.append(float(value))
            metric_rows.append(row)
        return np.asarray(metric_rows, dtype=float)

    def select(
        self,
        *,
        candidates_actual: np.ndarray,
        candidates_normalized: np.ndarray,
        history: list[dict[str, Any]],
        tr_debug: dict[str, Any],
    ) -> SelectionResult:
        if self._torch is None or self._device is None or self._surrogate is None:
            raise RuntimeError("BNN acquisition backend is not fitted")

        Xcand = np.asarray(candidates_actual, dtype=float)
        best_objective = min(float(row["objective"]) for row in history if np.isfinite(float(row["objective"])))
        candidate_dataset = self._candidate_dataset(Xcand)
        candidate_indices = np.arange(candidate_dataset.n_samples, dtype=int)
        _, response_samples = self._surrogate.predict_s_params(
            candidate_dataset,
            candidate_indices,
            mc_samples=int(self.config.mc_samples),
            return_samples=True,
        )
        metric_samples = self._metric_samples_from_response_samples(response_samples)
        improvement = np.maximum(float(best_objective) - metric_samples, 0.0)
        if self.config.acquisition == "qlogei":
            scores = np.log(np.maximum(improvement.mean(axis=0), np.finfo(float).tiny))
        elif self.config.acquisition in {"qei", "ei"}:
            scores = improvement.mean(axis=0)
        else:
            mean = metric_samples.mean(axis=0)
            std = metric_samples.std(axis=0)
            scores = -(mean - float(self.config.beta) * std)
        pos = int(np.argmax(scores))
        means = metric_samples.mean(axis=0)
        variances = metric_samples.var(axis=0)
        return SelectionResult(
            requested_geometry=np.asarray(candidates_actual[pos], dtype=float),
            predicted_mean=float(means.reshape(-1)[pos]),
            predicted_uncertainty=float(np.sqrt(max(float(variances.reshape(-1)[pos]), 0.0))),
            acquisition_score=float(scores[pos]),
            selection_debug={
                "mode": f"bnn_{self.config.acquisition}_discrete",
                "model_fit_status": "ok",
                "selected_pos": pos,
                "best_score": float(scores[pos]),
                "candidate_pool_size": int(len(candidates_actual)),
                **tr_debug,
                "bnn_runtime": {
                    "device": str(self._device),
                    "dtype": "torch.float32",
                    "x_is_normalized": False,
                    "surrogate": "response",
                    "surrogate_variant": str(self._surrogate.surrogate_variant),
                    "mc_samples": int(self.config.mc_samples),
                    "epochs": int(self.config.epochs),
                    "update_epochs": int(self.config.update_epochs),
                    "warm_start": bool(self.config.warm_start),
                    "fit_mode": self._last_fit_mode,
                    "train_count": int(self._last_train_count),
                    "trained_epochs": int(self._surrogate.trained_epochs),
                    "checkpoint_path": (
                        None if self._checkpoint_path() is None else str(self._checkpoint_path())
                    ),
                    "force_retrain": bool(self.config.force_retrain),
                },
            },
        )
