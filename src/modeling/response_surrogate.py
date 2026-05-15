from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .response_tcn import ResponseTCN, ResponseTCNConfig


@dataclass(frozen=True)
class ResponseTrainingSample:
    geometry_normalized: np.ndarray
    r: np.ndarray
    i: np.ndarray


@dataclass(frozen=True)
class ResponseSurrogatePrediction:
    r: np.ndarray
    i: np.ndarray


class ResponseSurrogateTrainer:
    def __init__(
        self,
        *,
        config: ResponseTCNConfig,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if lr <= 0:
            raise ValueError("lr must be > 0")
        self.config = config
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self.device_name = device

    def train(self, X: np.ndarray, r: np.ndarray, i: np.ndarray) -> ResponseTCN:
        import torch

        X_arr, r_arr, i_arr = validate_response_arrays(X, r, i, self.config)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        device = torch.device(self.device_name) if self.device_name is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = ResponseTCN(self.config).to(device)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(r_arr, dtype=torch.float32),
            torch.tensor(i_arr, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for xb, rb, ib in loader:
                xb = xb.to(device)
                rb = rb.to(device)
                ib = ib.to(device)
                outputs = model(xb)
                loss = loss_fn(outputs["r"], rb) + loss_fn(outputs["i"], ib)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model


class ResponseSurrogate:
    def __init__(
        self,
        *,
        model_config: ResponseTCNConfig,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.trainer = ResponseSurrogateTrainer(
            config=model_config,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
        )
        self._model = None
        self._device = None

    @classmethod
    def from_config_mapping(
        cls,
        values: Mapping[str, object],
        *,
        input_dim: int,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int = 0,
        device: str | None = None,
    ) -> "ResponseSurrogate":
        model_config = ResponseTCNConfig.from_mapping(values, input_dim=input_dim)
        return cls(
            model_config=model_config,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
        )

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, X: np.ndarray, r: np.ndarray, i: np.ndarray) -> None:
        import torch

        self._model = self.trainer.train(X, r, i)
        self._model.eval()
        self._device = next(self._model.parameters()).device
        # Keep Torch deterministic for subsequent dropout-free eval predictions.
        torch.manual_seed(self.trainer.seed)

    def predict(self, X: np.ndarray) -> ResponseSurrogatePrediction:
        import torch

        if self._model is None or self._device is None:
            raise RuntimeError("ResponseSurrogate is not fitted")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X_arr.shape[1] != self.model_config.input_dim:
            raise ValueError(f"X dimension mismatch: got {X_arr.shape[1]}, expected {self.model_config.input_dim}")

        with torch.no_grad():
            tensor = torch.tensor(X_arr, dtype=torch.float32, device=self._device)
            outputs = self._model(tensor)
        return ResponseSurrogatePrediction(
            r=outputs["r"].detach().cpu().numpy(),
            i=outputs["i"].detach().cpu().numpy(),
        )


def stack_training_samples(samples: list[ResponseTrainingSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("at least one training sample is required")
    X = np.asarray([sample.geometry_normalized for sample in samples], dtype=float)
    r = np.asarray([sample.r for sample in samples], dtype=float)
    i = np.asarray([sample.i for sample in samples], dtype=float)
    return X, r, i


def validate_response_arrays(
    X: np.ndarray,
    r: np.ndarray,
    i: np.ndarray,
    config: ResponseTCNConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    i_arr = np.asarray(i, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X_arr.shape[1] != config.input_dim:
        raise ValueError(f"X dimension mismatch: got {X_arr.shape[1]}, expected {config.input_dim}")
    if r_arr.shape != i_arr.shape:
        raise ValueError("r and i response arrays must have the same shape")
    expected_shape = (X_arr.shape[0], *config.output_channels["r"])
    if r_arr.shape != expected_shape:
        raise ValueError(f"response shape mismatch: got {r_arr.shape}, expected {expected_shape}")
    return X_arr, r_arr, i_arr
