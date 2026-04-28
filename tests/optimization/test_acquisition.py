from __future__ import annotations

import numpy as np

from optimization.acquisition import EIAcquisition, UCBAcquisition


def test_ucb_uses_uncertainty_and_falls_back_to_mean() -> None:
    mu = np.array([0.3, 0.4, 0.2], dtype=float)
    sigma = np.array([0.0, 0.2, 0.5], dtype=float)

    acq = UCBAcquisition(beta=1.5)
    score = acq.score(mu, sigma)
    score_no_unc = acq.score(mu, None)

    assert np.argmax(score) == 2
    assert np.allclose(score_no_unc, -mu)


def test_ei_prefers_positive_improvement() -> None:
    mu = np.array([0.1, 0.6, 0.3], dtype=float)
    sigma = np.array([0.2, 0.1, 0.5], dtype=float)
    best = 0.25  # minimization best-so-far

    acq = EIAcquisition(xi=0.0)
    score = acq.score(mu, sigma, best_y=best)

    assert score.shape == (3,)
    assert np.argmax(score) == 0


def test_ei_falls_back_to_mean_when_best_or_sigma_missing() -> None:
    mu = np.array([0.1, 0.2], dtype=float)

    acq = EIAcquisition()
    assert np.allclose(acq.score(mu, None, best_y=0.0), -mu)
    assert np.allclose(acq.score(mu, np.array([0.1, 0.1]), best_y=None), -mu)
