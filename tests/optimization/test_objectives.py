from __future__ import annotations

import numpy as np
import pytest

from optimization.constraints import GreaterEqualConstraint, LessEqualConstraint
from optimization.metrics import AtFrequencyMetric, BandMaxMetric, BandMeanMetric, BandMinMetric, PeakFrequencyMetric
from optimization.objectives import (
    LowerBoundHingePenalty,
    MaximizeMetric,
    MinimizeMetric,
    ObjectiveFunction,
    TargetMetric,
    Objective,
    UpperBoundHingePenalty,
    band_ghz_to_hz,
    ghz_to_hz,
)
from optimization.optimization_spec import BCLFOptimizationSpec, build_bclf_objective, build_bclf_optimization_spec
from optimization.sparam_response import SParameterResponse


def _sample_response(freq_len: int = 5) -> SParameterResponse:
    freqs = np.linspace(5.0e9, 6.0e9, freq_len)
    s11_db = np.full(freq_len, -10.0, dtype=float)
    s21_db = np.full(freq_len, -6.0, dtype=float)
    return SParameterResponse(freq_hz=freqs, s_db={"S11": s11_db, "S21": s21_db})


def _from_db(freqs: np.ndarray, s11_db: np.ndarray, s21_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_params = np.zeros((2, 2, len(freqs)), dtype=complex)
    s_params[0, 0, :] = 10 ** (np.asarray(s11_db, dtype=float) / 20.0)
    s_params[1, 0, :] = 10 ** (np.asarray(s21_db, dtype=float) / 20.0)
    return freqs, s_params


def test_basic_metrics_compute() -> None:
    resp = _sample_response()
    assert BandMaxMetric("S11").compute(resp) == pytest.approx(-10.0, abs=1e-6)
    assert BandMinMetric("S21").compute(resp) == pytest.approx(-6.0, abs=1e-6)


def test_objective_function_minimizes_terms() -> None:
    resp = _sample_response()
    obj = ObjectiveFunction(
        terms=[
            MaximizeMetric(name="max_s21", metric=BandMeanMetric("S21", (5.0e9, 6.0e9))),
            MinimizeMetric(name="min_s11_max", metric=BandMaxMetric("S11", (5.0e9, 6.0e9)), weight=2.0),
        ]
    )
    result = obj.evaluate(resp)

    assert result.value == pytest.approx(-14.0, abs=1e-6)
    assert result.objective == pytest.approx(-14.0, abs=1e-6)
    assert result.feasible is None
    assert "max_s21" in result.objective_components


def test_target_term_and_constraints_are_separated() -> None:
    resp = _sample_response()
    obj = ObjectiveFunction(
        terms=[
            TargetMetric(
                name="target_peak",
                metric=PeakFrequencyMetric("S21", (5.0e9, 6.0e9)),
                target=5.5e9,
                scale=0.5e9,
                weight=2.0,
            )
        ],
        constraints=[
            GreaterEqualConstraint(
                name="s21_mean_ge_-5",
                metric=BandMeanMetric("S21", (5.0e9, 6.0e9)),
                lower_bound=-5.0,
            ),
            LessEqualConstraint(
                name="s11_max_le_-12",
                metric=BandMaxMetric("S11", (5.0e9, 6.0e9)),
                upper_bound=-12.0,
            ),
        ],
    )
    result = obj.evaluate(resp)

    assert result.feasible is False
    assert result.constraint_violations["s21_mean_ge_-5"] == pytest.approx(1.0, abs=1e-6)
    assert result.constraint_violations["s11_max_le_-12"] == pytest.approx(2.0, abs=1e-6)
    assert result.metrics["penalty"] == 0.0


def test_raw_freqs_sparams_entrypoint() -> None:
    freqs = np.array([5.2e9, 5.5e9, 5.9e9])
    s11 = np.array([-12.0, -11.0, -10.0])
    s21 = np.array([-4.0, -3.0, -5.0])
    f, s = _from_db(freqs, s11, s21)

    obj = ObjectiveFunction(
        terms=[MaximizeMetric(name="center", metric=AtFrequencyMetric("S21", 5.5e9))]
    )
    out = obj.evaluate(f, s)
    assert out.value == pytest.approx(3.0, abs=1e-6)


def test_hinge_penalty_terms() -> None:
    resp = _sample_response()
    lower = LowerBoundHingePenalty(
        name="lower",
        metric=BandMeanMetric("S21", (5.0e9, 6.0e9)),
        bound=-5.0,
        weight=5.0,
    )
    upper = UpperBoundHingePenalty(
        name="upper",
        metric=BandMaxMetric("S11", (5.0e9, 6.0e9)),
        bound=-12.0,
        weight=5.0,
    )
    obj = ObjectiveFunction(terms=[lower, upper], constraints=[])
    out = obj.evaluate(resp)
    # mean S21=-6 -> hinge=1 ; max S11=-10 -> hinge=2 ; weighted by 5.
    assert out.value == pytest.approx(15.0, abs=1e-6)
    assert out.feasible is None
    assert out.constraint_violations == {}


def test_bclf_objective_ghz_builder_and_spec() -> None:
    freqs = np.array([3.6e9, 4.2e9, 5.2e9, 5.55e9, 5.9e9, 6.5e9, 7.5e9, 8.7e9])
    s11 = np.array([-12, -12, -9, -8, -10, -15, -15, -16], dtype=float)
    s21 = np.array([-30, -28, -6, -4, -7, -20, -22, -21], dtype=float)
    f, s = _from_db(freqs, s11, s21)

    obj = BCLFOptimizationSpec().to_objective()
    out = obj.evaluate(f, s)

    assert len(out.constraint_violations) == 0
    assert isinstance(out.value, float)

    spec = BCLFOptimizationSpec()
    assert spec.frequency_bands_ghz["passband"] == (5.2, 5.9)


def test_bclf_penalized_formula_matches_definition() -> None:
    freqs = np.array([3.6e9, 4.2e9, 5.2e9, 5.55e9, 5.9e9, 6.5e9, 7.5e9, 8.7e9])
    s11 = np.array([-12, -12, -9, -8, -10, -15, -15, -16], dtype=float)
    s21 = np.array([-30, -28, -6, -4, -7, -20, -22, -21], dtype=float)
    f, s = _from_db(freqs, s11, s21)

    obj = BCLFOptimizationSpec().to_objective()
    out = obj.evaluate(f, s)

    a_mean = np.mean(np.array([-6.0, -4.0, -7.0]))
    b_mean = np.mean(np.array([-9.0, -8.0, -10.0]))
    c_mean = np.mean(np.array([-30.0, -28.0, -20.0, -22.0, -21.0]))
    s_score = a_mean - 0.5 * b_mean - 0.5 * c_mean

    a_worst = -7.0
    b_worst = -8.0
    c_worst = -20.0
    p = max(0.0, -7.0 - a_worst) + max(0.0, b_worst + 7.0) + max(0.0, c_worst + 18.0)
    j = s_score - 5.0 * p

    # ObjectiveFunction minimizes, so returned objective is -J.
    assert out.value == pytest.approx(-j, abs=1e-6)
    assert out.constraint_metrics == {}
    assert out.constraint_violations == {}


def test_bclf_compatibility_wrappers() -> None:
    spec = build_bclf_optimization_spec()
    assert isinstance(spec, BCLFOptimizationSpec)
    obj = build_bclf_objective()
    assert isinstance(obj, Objective)


def test_unit_helpers() -> None:
    assert ghz_to_hz(5.55) == pytest.approx(5.55e9)
    assert band_ghz_to_hz((3.6, 4.8)) == (3.6e9, 4.8e9)


def test_sparam_response_shape_errors() -> None:
    freqs = np.array([1.0, 2.0, 3.0])
    bad = np.zeros((2, 3), dtype=complex)
    with pytest.raises(ValueError, match="3D array"):
        ObjectiveFunction([MaximizeMetric(name="m", metric=BandMeanMetric("S21", (1, 3)))]).evaluate(freqs, bad)

    s = np.zeros((2, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="does not match"):
        ObjectiveFunction([MaximizeMetric(name="m", metric=BandMeanMetric("S21", (1, 3)))]).evaluate(freqs, s)
