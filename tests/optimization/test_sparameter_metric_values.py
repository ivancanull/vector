from __future__ import annotations

import numpy as np
import pytest

from optimization.metrics import (
    DifferentialInductanceAtFrequencyMetric,
    DifferentialQAtFrequencyMetric,
    DifferentialSRFMetric,
    SParameterAtFrequencyMetric,
    compute_sparameter_metric_values,
)
from optimization.sparam_response import SParameterResponse


def _inductor_sparams(freqs: np.ndarray, l_nh: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    z0 = 50.0
    s_params = np.zeros((2, 2, len(freqs)), dtype=complex)
    for idx, freq in enumerate(freqs):
        omega = 2.0 * np.pi * float(freq)
        x = omega * float(l_nh[idx]) * 1e-9
        r = x / float(q_values[idx])
        zdiff = r + 1j * x
        z_matrix = np.asarray([[zdiff / 2.0, 0.0], [0.0, zdiff / 2.0]], dtype=complex)
        ident = np.eye(2, dtype=complex)
        s_matrix = (z_matrix - z0 * ident) @ np.linalg.inv(z_matrix + z0 * ident)
        s_params[:, :, idx] = s_matrix
    return s_params


def test_sparameter_metric_values_extract_point_and_differential_metrics() -> None:
    freqs = np.array([4.0e9, 5.0e9, 6.0e9])
    response = SParameterResponse.from_sparams(
        freqs,
        _inductor_sparams(
            freqs,
            l_nh=np.array([1.0, 1.2, -0.5]),
            q_values=np.array([10.0, 12.0, -10.0]),
        ),
    )
    values = compute_sparameter_metric_values(
        response,
        [
            DifferentialInductanceAtFrequencyMetric(5.0e9),
            DifferentialQAtFrequencyMetric(5.0e9),
            DifferentialSRFMetric(),
            SParameterAtFrequencyMetric("S11", 5.0e9),
            SParameterAtFrequencyMetric("S21", 5.0e9),
        ],
    )

    assert values["L_5GHz_nH"] == pytest.approx(1.2, abs=1e-6)
    assert values["Q_5GHz"] == pytest.approx(12.0, abs=1e-6)
    assert values["SRF_GHz"] == pytest.approx(5.6666666667, abs=1e-6)
    assert values["S11_5GHz_dB"] is not None
    assert values["S21_5GHz_dB"] is not None


def test_sparameter_response_accepts_skrf_frequency_first_shape() -> None:
    freqs = np.array([4.0e9, 5.0e9, 6.0e9])
    ports_ports_freq = _inductor_sparams(
        freqs,
        l_nh=np.array([1.0, 1.2, 1.3]),
        q_values=np.array([10.0, 12.0, 13.0]),
    )
    freq_ports_ports = np.moveaxis(ports_ports_freq, 2, 0)

    response = SParameterResponse.from_sparams(freqs, freq_ports_ports)

    assert response.s_params is not None
    assert response.s_params.shape == (2, 2, 3)
