from __future__ import annotations

import numpy as np
import pytest

from optimization.constraints import GreaterEqualConstraint, LessEqualConstraint
from optimization.metrics import AtFrequencyMetric, BandMaxMetric, BandMeanMetric, PeakFrequencyMetric
from optimization.objectives import (
    BCLFObjective,
    MaxS11DbMetric,
    MaximizeMetric,
    MinS21DbMetric,
    MinimizeMetric,
    ObjectiveFunction,
    TargetMetric,
    build_bclf_objective,
)
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
    assert MaxS11DbMetric().compute(resp) == pytest.approx(-10.0, abs=1e-6)
    assert MinS21DbMetric().compute(resp) == pytest.approx(-6.0, abs=1e-6)


def test_objective_function_minimizes_terms() -> None:
    resp = _sample_response()
    obj = ObjectiveFunction(
        terms=[
            MaximizeMetric(name="max_s21", metric=BandMeanMetric("S21", (5.0e9, 6.0e9))),
            MinimizeMetric(name="min_s11_max", metric=BandMaxMetric("S11", (5.0e9, 6.0e9)), weight=2.0),
        ]
    )
    result = obj.evaluate(resp)

    # max S21 => -(-6)=+6 ; min S11 max => 2*(-10)=-20 ; total=-14 (minimize)
    assert result.value == pytest.approx(-14.0, abs=1e-6)
    assert result.objective == pytest.approx(-14.0, abs=1e-6)
    assert result.feasible is True


def test_target_term_and_constraints_penalty() -> None:
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
        penalty_weight=100.0,
    )
    result = obj.evaluate(resp)

    # s21_mean=-6 -> violation=1, s11_max=-10 -> violation=2 => total=3, penalty=300
    assert result.feasible is False
    assert result.metrics["total_positive_violation"] == pytest.approx(3.0, abs=1e-6)
    assert result.metrics["penalty"] == pytest.approx(300.0, abs=1e-6)
    assert result.value > 200.0


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


def test_bclf_objective_and_builder() -> None:
    freqs = np.array([3.6e9, 4.2e9, 5.2e9, 5.55e9, 5.9e9, 6.5e9, 7.5e9, 8.7e9])
    s11 = np.array([-12, -12, -9, -8, -10, -15, -15, -16], dtype=float)
    s21 = np.array([-30, -28, -6, -4, -7, -20, -22, -21], dtype=float)
    f, s = _from_db(freqs, s11, s21)

    obj = build_bclf_objective()
    out = obj.evaluate(f, s)

    assert "passband_s21_mean_geq" in " ".join(out.constraints.keys()) or len(out.constraints) == 4
    assert "penalty" in out.metrics

    obj2 = BCLFObjective(
        passband_hz=(5.2e9, 5.9e9),
        lower_stopband_hz=(3.6e9, 4.8e9),
        upper_stopband_hz=(6.5e9, 8.7e9),
        target_center_hz=5.55e9,
        min_passband_s21_db=-5.0,
        max_passband_s11_db=-10.0,
        max_stopband_s21_db=-25.0,
        center_weight=2.0,
    )
    out2 = obj2.evaluate(f, s)
    assert isinstance(out2.value, float)


def test_sparam_response_shape_errors() -> None:
    freqs = np.array([1.0, 2.0, 3.0])
    bad = np.zeros((2, 3), dtype=complex)
    with pytest.raises(ValueError, match="3D array"):
        ObjectiveFunction([MaximizeMetric(name="m", metric=BandMeanMetric("S21", (1, 3)))]).evaluate(freqs, bad)

    s = np.zeros((2, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="does not match"):
        ObjectiveFunction([MaximizeMetric(name="m", metric=BandMeanMetric("S21", (1, 3)))]).evaluate(freqs, s)
