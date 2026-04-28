from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_pool: np.ndarray) -> np.ndarray:
        """Return mean prediction."""
        raise NotImplementedError

    @abstractmethod
    def predict_uncertainty(self, X_pool: np.ndarray) -> np.ndarray | None:
        """Return predictive uncertainty if available."""
        raise NotImplementedError


class _Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        values = np.asarray(x, dtype=float)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer is not fitted")
        values = np.asarray(x, dtype=float)
        return (values - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer is not fitted")
        values = np.asarray(x, dtype=float)
        return values * self.std_ + self.mean_


class _TorchMLPBase(SurrogateModel):
    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        lr: float = 1e-2,
        epochs: int = 120,
        batch_size: int = 32,
        seed: int = 0,
        dropout: float = 0.0,
        mc_samples: int = 32,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.dropout = dropout
        self.mc_samples = mc_samples

        self._fitted = False
        self._x_scaler = _Standardizer()
        self._y_scaler = _Standardizer()
        self._model = None
        self._device = None

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn

        layers: list[nn.Module] = []
        prev = input_dim
        for width in self.hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        import torch

        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D array")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X_train and y_train size mismatch")

        torch.manual_seed(self.seed)
        self._x_scaler.fit(X)
        self._y_scaler.fit(y)
        Xn = self._x_scaler.transform(X)
        yn = self._y_scaler.transform(y)

        X_t = torch.tensor(Xn, dtype=torch.float32)
        y_t = torch.tensor(yn, dtype=torch.float32)

        self._device = torch.device("cpu")
        self._model = self._build_model(X.shape[1]).to(self._device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._fitted = True

    def _predict_raw(self, X_pool: np.ndarray) -> np.ndarray:
        import torch

        if not self._fitted or self._model is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X_pool, dtype=float)
        Xn = self._x_scaler.transform(X)
        X_t = torch.tensor(Xn, dtype=torch.float32)
        with torch.no_grad():
            out = self._model(X_t).cpu().numpy().reshape(-1, 1)
        return self._y_scaler.inverse_transform(out).reshape(-1)

    def predict(self, X_pool: np.ndarray) -> np.ndarray:
        return self._predict_raw(X_pool)


class ANNSurrogate(_TorchMLPBase):
    def predict_uncertainty(self, X_pool: np.ndarray) -> np.ndarray | None:
        _ = X_pool
        return None


class BNNSurrogate(_TorchMLPBase):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dropout", 0.1)
        super().__init__(**kwargs)

    def _mc_stats(self, X_pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if not self._fitted or self._model is None:
            raise RuntimeError("Model is not fitted")

        X = np.asarray(X_pool, dtype=float)
        Xn = self._x_scaler.transform(X)
        X_t = torch.tensor(Xn, dtype=torch.float32)

        preds = []
        self._model.train()
        with torch.no_grad():
            for _ in range(self.mc_samples):
                out = self._model(X_t).cpu().numpy().reshape(-1, 1)
                y = self._y_scaler.inverse_transform(out).reshape(-1)
                preds.append(y)
        stacked = np.asarray(preds, dtype=float)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def predict(self, X_pool: np.ndarray) -> np.ndarray:
        mean, _ = self._mc_stats(X_pool)
        return mean

    def predict_uncertainty(self, X_pool: np.ndarray) -> np.ndarray | None:
        _, std = self._mc_stats(X_pool)
        return std


class GPSurrogate(SurrogateModel):
    """Lightweight RBF-kernel GP regressor (no extra dependencies)."""

    def __init__(
        self,
        *,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-6,
    ) -> None:
        self.length_scale = float(length_scale)
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)

        self._fitted = False
        self._x_scaler = _Standardizer()
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_mean: float | None = None
        self._chol: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return self.signal_variance * np.exp(-0.5 * sqdist / (self.length_scale ** 2))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X_train must be a 2D array")
        if len(y) != len(X):
            raise ValueError("X_train and y_train size mismatch")

        self._x_scaler.fit(X)
        Xn = self._x_scaler.transform(X)

        self._X_train = Xn
        self._y_mean = float(y.mean())
        self._y_train = y - self._y_mean

        K = self._rbf_kernel(Xn, Xn)
        K += np.eye(len(Xn), dtype=float) * self.noise_variance

        # Cholesky solves are numerically stable for GP posterior updates.
        self._chol = np.linalg.cholesky(K)
        z = np.linalg.solve(self._chol, self._y_train)
        self._alpha = np.linalg.solve(self._chol.T, z)
        self._fitted = True

    def predict(self, X_pool: np.ndarray) -> np.ndarray:
        if not self._fitted or self._X_train is None or self._alpha is None or self._y_mean is None:
            raise RuntimeError("Model is not fitted")

        X = np.asarray(X_pool, dtype=float)
        Xn = self._x_scaler.transform(X)
        K_star = self._rbf_kernel(Xn, self._X_train)
        return (K_star @ self._alpha) + self._y_mean

    def predict_uncertainty(self, X_pool: np.ndarray) -> np.ndarray | None:
        if not self._fitted or self._X_train is None or self._chol is None:
            raise RuntimeError("Model is not fitted")

        X = np.asarray(X_pool, dtype=float)
        Xn = self._x_scaler.transform(X)
        K_star = self._rbf_kernel(Xn, self._X_train)
        v = np.linalg.solve(self._chol, K_star.T)
        var = self.signal_variance - np.sum(v * v, axis=0)
        var = np.maximum(var, 0.0)
        return np.sqrt(var)
