from __future__ import annotations

import numpy as np
import pytest

from modeling import ResponseSurrogate, ResponseTCNConfig


def _responses(n: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.linspace(0.0, 1.0, 401, dtype=float)
    r = np.stack([np.stack([base, base * 0.5], axis=0) for _ in range(n)], axis=0)
    i = np.stack([np.stack([base * 0.1, base * 0.2], axis=0) for _ in range(n)], axis=0)
    return r, i


def test_response_surrogate_tiny_training_predicts_finite_values() -> None:
    config = ResponseTCNConfig(input_dim=2, dropout=0.0)
    surrogate = ResponseSurrogate(model_config=config, epochs=1, batch_size=2, lr=1e-3, seed=3, device="cpu")
    X = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    r, i = _responses(2)

    surrogate.fit(X, r, i)
    pred = surrogate.predict(np.asarray([[0.5, 0.5]], dtype=float))

    assert pred.r.shape == (1, 2, 401)
    assert pred.i.shape == (1, 2, 401)
    assert np.all(np.isfinite(pred.r))
    assert np.all(np.isfinite(pred.i))


def test_response_surrogate_rejects_batch_mismatch() -> None:
    config = ResponseTCNConfig(input_dim=2, dropout=0.0)
    surrogate = ResponseSurrogate(model_config=config, epochs=1, device="cpu")
    r, i = _responses(1)

    with pytest.raises(ValueError, match="response shape mismatch"):
        surrogate.fit(np.zeros((2, 2), dtype=float), r, i)
