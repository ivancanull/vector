from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .bclf_dataset import BCLFResponseDataset, response_db

try:
    from src.bayesian.modules.loss import BKLLoss
    from src.bayesian.modules.models import MultiOutputBNN
    from src.regularizer import MinMaxRegularizer, NoneRegularizer
    from src.train import train as legacy_train
except ImportError:  # pragma: no cover - pytest uses src/ as pythonpath
    from bayesian.modules.loss import BKLLoss
    from bayesian.modules.models import MultiOutputBNN
    from regularizer import MinMaxRegularizer, NoneRegularizer

    legacy_train = None


DEFAULT_PORT_COMBINATIONS = [(0, 0), (0, 1)]
DEFAULT_BCLF_INPUT_MIN = [7.7, 8.0, 7.8, 0.9, 0.7, 0.9, 0.13, 0.55, 3.5e9]
DEFAULT_BCLF_INPUT_MAX = [7.9, 8.2, 8.0, 1.1, 0.9, 1.1, 0.16, 0.6, 8.7e9]
DEFAULT_FOURIER_BANDS = [1.0, 2.0, 4.0, 8.0, 16.0]


class _FlattenedResponseDataset:
    def __init__(self, X: np.ndarray, targets: dict[str, np.ndarray]) -> None:
        import torch

        self.X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.targets = {
            name: torch.tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32)
            for name, value in targets.items()
        }

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], {name: value[idx] for name, value in self.targets.items()}


class _TensorStandardizer:
    def __init__(self, values: np.ndarray, *, eps: float = 1e-8) -> None:
        import torch

        arr = np.asarray(values, dtype=np.float32)
        self.mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        std = arr.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        self.std = torch.tensor(std, dtype=torch.float32)

    def regularize(self, n):
        self.mean = self.mean.to(n.device)
        self.std = self.std.to(n.device)
        return (n - self.mean) / self.std

    def deregularize(self, n):
        self.mean = self.mean.to(n.device)
        self.std = self.std.to(n.device)
        return n * self.std + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "_TensorStandardizer":
        obj = cls.__new__(cls)
        import torch

        obj.mean = torch.tensor(payload["mean"], dtype=torch.float32)
        obj.std = torch.tensor(payload["std"], dtype=torch.float32)
        return obj


class LegacyBNNResponseSurrogate:
    """Response surrogate using the legacy MultiOutputBNN Bayesian layers."""

    def __init__(
        self,
        *,
        input_bounds_min: list[float] | None = None,
        input_bounds_max: list[float] | None = None,
        hidden_dims: list[int] | None = None,
        prior_sigma: float = 0.1,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        bayesian_loss_weight: float = 0.01,
        mc_samples: int = 100,
        standardize_targets: bool = True,
        target_mode: str = "ri",
        seed: int = 0,
        device: str | None = None,
        port_combinations: list[tuple[int, int]] | None = None,
    ) -> None:
        self.input_bounds_min = list(input_bounds_min or DEFAULT_BCLF_INPUT_MIN)
        self.input_bounds_max = list(input_bounds_max or DEFAULT_BCLF_INPUT_MAX)
        if len(self.input_bounds_min) != len(self.input_bounds_max):
            raise ValueError("input_bounds_min and input_bounds_max must have the same length")
        self.hidden_dims = list(hidden_dims or [256, 256, 256])
        self.prior_sigma = float(prior_sigma)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.bayesian_loss_weight = float(bayesian_loss_weight)
        self.mc_samples = int(mc_samples)
        self.standardize_targets = bool(standardize_targets)
        if target_mode not in {"ri", "db"}:
            raise ValueError("target_mode must be 'ri' or 'db'")
        self.target_mode = target_mode
        self.seed = int(seed)
        self.device_name = device
        self.port_combinations = list(port_combinations or DEFAULT_PORT_COMBINATIONS)
        self.model = None
        self._device = None
        self._regularizer = None
        self._optimizer = None
        self.trained_epochs = 0
        self.surrogate_variant = "raw"
        self.checkpoint_metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def fit(self, dataset: BCLFResponseDataset, train_indices: Iterable[int], *, epochs: int | None = None) -> None:
        import torch

        indices = np.asarray(list(train_indices), dtype=int)
        if indices.size == 0:
            raise ValueError("train_indices must be non-empty")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        X_flat, targets = self._build_flat_training_arrays(dataset, indices)
        self._device = torch.device(self.device_name) if self.device_name else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model()
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._regularizer = (
            MinMaxRegularizer(self.input_bounds_min, self.input_bounds_max),
            self._build_target_regularizers(targets),
        )
        self._train_arrays(X_flat, targets, epochs=int(epochs or self.epochs), generator_seed=self.seed)
        self.trained_epochs = int(epochs or self.epochs)

    def update(self, dataset: BCLFResponseDataset, train_indices: Iterable[int], *, epochs: int) -> None:
        """Continue training the existing model on the provided enlarged index set."""

        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.model is None or self._regularizer is None or self._device is None:
            raise RuntimeError("LegacyBNNResponseSurrogate must be fitted before update")
        if self._optimizer is None:
            import torch

            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        indices = np.asarray(list(train_indices), dtype=int)
        if indices.size == 0:
            raise ValueError("train_indices must be non-empty")
        X_flat, targets = self._build_flat_training_arrays(dataset, indices)
        self._train_arrays(X_flat, targets, epochs=int(epochs), generator_seed=self.seed + self.trained_epochs)
        self.trained_epochs += int(epochs)

    def _train_arrays(
        self,
        X_flat: np.ndarray,
        targets: dict[str, np.ndarray],
        *,
        epochs: int,
        generator_seed: int,
    ) -> None:
        import torch

        if self.model is None or self._optimizer is None or self._regularizer is None or self._device is None:
            raise RuntimeError("model, optimizer, regularizer, and device must be initialized")
        loader = torch.utils.data.DataLoader(
            _FlattenedResponseDataset(X_flat, targets),
            batch_size=min(self.batch_size, len(X_flat)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(generator_seed)),
        )
        loss_fns = {name: torch.nn.MSELoss() for name in targets}
        loss_weights = {name: 1.0 for name in targets}
        train_fn = legacy_train or _legacy_train
        self.model, _ = train_fn(
            epochs=int(epochs),
            dataloader=loader,
            model=self.model,
            optimizer=self._optimizer,
            data_loss_fns=loss_fns,
            data_loss_weights=loss_weights,
            bayesian_loss_fn=BKLLoss(reduction="mean", last_layer_only=False),
            bayesian_loss_weight=self.bayesian_loss_weight,
            regularizer=self._regularizer,
            device=self._device,
        )

    def _build_model(self):
        prior_mus = [0.0] * len(self.hidden_dims)
        prior_sigmas = [self.prior_sigma] * len(self.hidden_dims)
        return MultiOutputBNN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            prior_mus=prior_mus,
            prior_sigmas=prior_sigmas,
            output_channels=self._output_channels(),
        )

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        import torch

        if self.model is None or self._regularizer is None:
            raise RuntimeError("LegacyBNNResponseSurrogate is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": None if self._optimizer is None else self._optimizer.state_dict(),
            "config": self.config_dict(),
            "target_regularizers": self._target_regularizer_state(),
            "trained_epochs": int(self.trained_epochs),
            "metadata": dict(metadata or {}),
        }
        self.checkpoint_metadata = dict(metadata or {})
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "LegacyBNNResponseSurrogate":
        import torch

        payload = torch.load(path, map_location=torch.device(device or "cpu"))
        config = dict(payload["config"])
        variant = str(config.pop("surrogate_variant", "raw"))
        if cls is LegacyBNNResponseSurrogate and variant != "raw":
            raise ValueError(
                f"Checkpoint variant {variant!r} requires loading with the matching surrogate class"
            )
        if cls is not LegacyBNNResponseSurrogate and variant != cls.surrogate_variant_name():
            raise ValueError(
                f"Checkpoint variant {variant!r} does not match loader {cls.__name__}"
            )
        if device is not None:
            config["device"] = device
        obj = cls(**config)
        obj.model = obj._build_model()
        obj._device = torch.device(obj.device_name) if obj.device_name else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        obj.model.load_state_dict(payload["model_state_dict"])
        obj.model.to(obj._device)
        obj._optimizer = torch.optim.Adam(obj.model.parameters(), lr=obj.lr)
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            obj._optimizer.load_state_dict(optimizer_state)
        obj._regularizer = (
            MinMaxRegularizer(obj.input_bounds_min, obj.input_bounds_max),
            obj._target_regularizers_from_state(payload["target_regularizers"]),
        )
        obj.trained_epochs = int(payload.get("trained_epochs", obj.epochs))
        obj.checkpoint_metadata = dict(payload.get("metadata", {}) or {})
        return obj

    @property
    def input_dim(self) -> int:
        return len(self.input_bounds_min)

    @classmethod
    def surrogate_variant_name(cls) -> str:
        return "raw"

    def config_dict(self) -> dict[str, Any]:
        return {
            "surrogate_variant": self.surrogate_variant,
            "input_bounds_min": list(self.input_bounds_min),
            "input_bounds_max": list(self.input_bounds_max),
            "hidden_dims": list(self.hidden_dims),
            "prior_sigma": self.prior_sigma,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "bayesian_loss_weight": self.bayesian_loss_weight,
            "mc_samples": self.mc_samples,
            "standardize_targets": self.standardize_targets,
            "target_mode": self.target_mode,
            "seed": self.seed,
            "device": self.device_name,
            "port_combinations": [tuple(item) for item in self.port_combinations],
        }

    def predict_s_params(
        self,
        dataset: BCLFResponseDataset,
        candidate_indices: Iterable[int],
        *,
        mc_samples: int | None = None,
        return_samples: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        import torch

        if self.model is None or self._device is None or self._regularizer is None:
            raise RuntimeError("LegacyBNNResponseSurrogate is not fitted")

        indices = np.asarray(list(candidate_indices), dtype=int)
        X_flat = self._build_flat_inputs(dataset, indices)
        tensor = torch.tensor(X_flat.astype(np.float32), dtype=torch.float32, device=self._device)
        tensor = self._regularizer[0].regularize(tensor)
        samples = int(mc_samples or self.mc_samples)

        # Keep BayesianLinear sampling active for MC uncertainty, but disable dropout.
        # Per-frequency dropout masks make S-parameter curves visibly jagged because
        # each frequency row is perturbed independently.
        self.model.eval()
        self.model.unfreeze()
        pred_samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(samples):
                out = self.model(tensor)
                if self.target_mode == "db":
                    db = self._regularizer[1]["db"].deregularize(out["db"]).detach().cpu().numpy()
                    pred_samples.append(self._flat_db_to_sparams(db, len(indices), dataset.n_freqs))
                else:
                    r = self._regularizer[1]["r"].deregularize(out["r"]).detach().cpu().numpy()
                    i = self._regularizer[1]["i"].deregularize(out["i"]).detach().cpu().numpy()
                    pred_samples.append(self._flat_ri_to_sparams(r, i, len(indices), dataset.n_freqs))
        stacked = np.asarray(pred_samples, dtype=complex)
        mean = stacked.mean(axis=0)
        if return_samples:
            return mean, stacked
        return mean

    def predict_db(
        self,
        dataset: BCLFResponseDataset,
        candidate_indices: Iterable[int],
        *,
        mc_samples: int | None = None,
    ) -> np.ndarray:
        return response_db(self.predict_s_params(dataset, candidate_indices, mc_samples=mc_samples), 0, 0)

    def _build_flat_training_arrays(
        self,
        dataset: BCLFResponseDataset,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        X_flat = self._build_flat_inputs(dataset, indices)
        if self.target_mode == "db":
            db_parts: list[np.ndarray] = []
            for idx in indices:
                selected = np.stack(
                    [response_db(dataset.s_params[idx], row, col) for row, col in self.port_combinations],
                    axis=1,
                )
                db_parts.append(selected)
            return X_flat, {"db": np.vstack(db_parts)}

        r_parts: list[np.ndarray] = []
        i_parts: list[np.ndarray] = []
        for idx in indices:
            selected = np.stack(
                [dataset.s_params[idx, row, col, :] for row, col in self.port_combinations],
                axis=1,
            )
            r_parts.append(np.real(selected))
            i_parts.append(np.imag(selected))
        return X_flat, {"r": np.vstack(r_parts), "i": np.vstack(i_parts)}

    def _build_flat_inputs(self, dataset: BCLFResponseDataset, indices: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        frequency_features = self._frequency_features(dataset.freqs)
        for idx in indices:
            geom = np.repeat(dataset.X[idx].reshape(1, -1), dataset.n_freqs, axis=0)
            rows.append(np.column_stack([geom, frequency_features]))
        return np.vstack(rows)

    def _frequency_features(self, freqs_hz: np.ndarray) -> np.ndarray:
        return np.asarray(freqs_hz, dtype=float).reshape(-1, 1)

    def _output_channels(self) -> dict[str, tuple[int, Any]]:
        if self.target_mode == "db":
            return {"db": (len(self.port_combinations), None)}
        return {"r": (len(self.port_combinations), None), "i": (len(self.port_combinations), None)}

    def _build_target_regularizers(self, targets: dict[str, np.ndarray]) -> dict[str, Any]:
        regularizers = {}
        for name, values in targets.items():
            regularizers[name] = _TensorStandardizer(values) if self.standardize_targets else NoneRegularizer()
        return regularizers

    def _flat_ri_to_sparams(self, r: np.ndarray, i: np.ndarray, n_cases: int, n_freqs: int) -> np.ndarray:
        response = (np.asarray(r, dtype=float) + 1j * np.asarray(i, dtype=float)).reshape(
            n_cases, n_freqs, len(self.port_combinations)
        )
        s_params = np.zeros((n_cases, 2, 2, n_freqs), dtype=complex)
        for channel_idx, (row, col) in enumerate(self.port_combinations):
            s_params[:, row, col, :] = response[:, :, channel_idx]
            if (row, col) == (0, 1):
                s_params[:, 1, 0, :] = response[:, :, channel_idx]
        return s_params

    def _flat_db_to_sparams(self, db: np.ndarray, n_cases: int, n_freqs: int) -> np.ndarray:
        response_db_values = np.asarray(db, dtype=float).reshape(n_cases, n_freqs, len(self.port_combinations))
        response = 10.0 ** (response_db_values / 20.0)
        s_params = np.zeros((n_cases, 2, 2, n_freqs), dtype=complex)
        for channel_idx, (row, col) in enumerate(self.port_combinations):
            s_params[:, row, col, :] = response[:, :, channel_idx]
            if (row, col) == (0, 1):
                s_params[:, 1, 0, :] = response[:, :, channel_idx]
        return s_params

    def _target_regularizer_state(self) -> dict[str, dict[str, Any] | None]:
        assert self._regularizer is not None
        payload = {}
        for name, regularizer in self._regularizer[1].items():
            payload[name] = regularizer.state_dict() if isinstance(regularizer, _TensorStandardizer) else None
        return payload

    def _target_regularizers_from_state(self, payload: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
        regularizers = {}
        for name in self._output_channels().keys():
            state = payload.get(name)
            regularizers[name] = _TensorStandardizer.from_state_dict(state) if state else NoneRegularizer()
        return regularizers


class FourierBNNResponseSurrogate(LegacyBNNResponseSurrogate):
    """Legacy BNN response surrogate with Fourier-encoded frequency features."""

    def __init__(
        self,
        *,
        fourier_bands: list[float] | None = None,
        include_raw_frequency: bool = True,
        frequency_min_hz: float = DEFAULT_BCLF_INPUT_MIN[-1],
        frequency_max_hz: float = DEFAULT_BCLF_INPUT_MAX[-1],
        input_bounds_min: list[float] | None = None,
        input_bounds_max: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.fourier_bands = [float(v) for v in (fourier_bands or DEFAULT_FOURIER_BANDS)]
        if not self.fourier_bands:
            raise ValueError("fourier_bands must be non-empty")
        self.include_raw_frequency = bool(include_raw_frequency)
        self.frequency_min_hz = float(frequency_min_hz)
        self.frequency_max_hz = float(frequency_max_hz)
        if self.frequency_min_hz >= self.frequency_max_hz:
            raise ValueError("frequency_min_hz must be < frequency_max_hz")

        geometry_min = list(input_bounds_min or DEFAULT_BCLF_INPUT_MIN[:8])
        geometry_max = list(input_bounds_max or DEFAULT_BCLF_INPUT_MAX[:8])
        if len(geometry_min) != 8 or len(geometry_max) != 8:
            raise ValueError("FourierBNNResponseSurrogate expects 8 geometry input bounds")

        encoded_min = list(geometry_min)
        encoded_max = list(geometry_max)
        if self.include_raw_frequency:
            encoded_min.append(0.0)
            encoded_max.append(1.0)
        encoded_min.extend([-1.0] * (2 * len(self.fourier_bands)))
        encoded_max.extend([1.0] * (2 * len(self.fourier_bands)))
        super().__init__(input_bounds_min=encoded_min, input_bounds_max=encoded_max, **kwargs)
        self.surrogate_variant = self.surrogate_variant_name()

    @classmethod
    def surrogate_variant_name(cls) -> str:
        return "fourier"

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        # Store only geometry bounds; Fourier encoded bounds are reconstructed.
        config["input_bounds_min"] = list(self.input_bounds_min[:8])
        config["input_bounds_max"] = list(self.input_bounds_max[:8])
        config["fourier_bands"] = list(self.fourier_bands)
        config["include_raw_frequency"] = bool(self.include_raw_frequency)
        config["frequency_min_hz"] = float(self.frequency_min_hz)
        config["frequency_max_hz"] = float(self.frequency_max_hz)
        return config

    def _frequency_features(self, freqs_hz: np.ndarray) -> np.ndarray:
        freq = np.asarray(freqs_hz, dtype=float).reshape(-1)
        f_norm = (freq - self.frequency_min_hz) / (self.frequency_max_hz - self.frequency_min_hz)
        columns: list[np.ndarray] = []
        if self.include_raw_frequency:
            columns.append(f_norm)
        for band in self.fourier_bands:
            phase = 2.0 * np.pi * float(band) * f_norm
            columns.append(np.sin(phase))
            columns.append(np.cos(phase))
        return np.column_stack(columns)


def _legacy_train(
    *,
    epochs: int,
    dataloader: Any,
    model: Any,
    optimizer: Any,
    data_loss_fns: dict[str, Any],
    data_loss_weights: dict[str, float],
    bayesian_loss_fn: Any,
    bayesian_loss_weight: float,
    regularizer: Any,
    device: Any,
) -> tuple[Any, Any]:
    from tqdm import tqdm

    model.train()
    model.to(device)
    model.unfreeze()
    with tqdm(total=epochs, unit="epoch") as pbar:
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in dataloader:
                inputs = regularizer[0].regularize(inputs.to(device))
                for key in targets.keys():
                    targets[key] = regularizer[1][key].regularize(targets[key].to(device))
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = 0.0
                if bayesian_loss_fn is not None:
                    loss = loss + bayesian_loss_weight * bayesian_loss_fn(model)
                for key in model.names:
                    loss = loss + data_loss_weights[key] * data_loss_fns[key](outputs[key], targets[key])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            if epoch % 10 == 9:
                average_loss = total_loss / max(len(dataloader), 1)
                pbar.set_postfix_str(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")
                pbar.update(10)
    return model, optimizer
