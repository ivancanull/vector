from __future__ import annotations

import numpy as np

from optimization.surrogates import ANNSurrogate, BNNSurrogate, GPSurrogate


def _toy_regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=float,
    )
    y_train = X_train[:, 0] + 2.0 * X_train[:, 1]
    X_test = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=float)
    return X_train, y_train, X_test


def test_ann_surrogate_contract() -> None:
    X_train, y_train, X_test = _toy_regression_data()
    model = ANNSurrogate(epochs=80, seed=7)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    unc = model.predict_uncertainty(X_test)

    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert unc is None


def test_bnn_surrogate_contract() -> None:
    X_train, y_train, X_test = _toy_regression_data()
    model = BNNSurrogate(epochs=80, seed=7, mc_samples=16)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    unc = model.predict_uncertainty(X_test)

    assert pred.shape == (2,)
    assert unc is not None
    assert unc.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert np.all(unc >= 0.0)


def test_gp_surrogate_contract_and_training_point_fit() -> None:
    X_train, y_train, X_test = _toy_regression_data()
    model = GPSurrogate(length_scale=0.8, noise_variance=1e-8)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred = model.predict(X_test)
    unc = model.predict_uncertainty(X_test)

    assert pred.shape == (2,)
    assert unc is not None
    assert unc.shape == (2,)
    assert np.all(unc >= 0.0)

    # GP should interpolate close to train targets when noise is tiny.
    assert np.max(np.abs(pred_train - y_train)) < 1e-2
