from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

__all__ = ["basic_poles_initialize"]


def basic_poles_initialize(freqs_range: Tuple, n_poles_real: int, n_poles_complex: int, scale="log"):
    """
    Basic poles initializer for Vector Fitting algorithm. The poles are initialized based on the frequency range with given sampling method.

    Args:

    - freqs_range: A tuple representing the frequency range (fmin, fmax).
    - n_poles_real: Number of real poles.
    - n_poles_complex: Number of complex poles conjugate pairs.

    Reference:
    https://github.com/scikit-rf/scikit-rf/blob/master/skrf/vectorFitting.py

    """
    fmin, fmax = freqs_range
    if scale == "log":
        pole_freqs_real = np.geomspace(fmin, fmax, n_poles_real)
        pole_freqs_complex = np.geomspace(fmin, fmax, n_poles_complex)
    elif scale == "linear":
        pole_freqs_real = np.linspace(fmin, fmax, n_poles_real)
        pole_freqs_complex = np.linspace(fmin, fmax, n_poles_complex)
    
    else:
        raise ValueError("Invalid scale. Choose 'log' or 'linear'.")

    poles = np.zeros(n_poles_real + 2 * n_poles_complex, dtype=complex)
    poles[:n_poles_real] = -pole_freqs_real * 2 * np.pi
    poles[n_poles_real::2] = pole_freqs_complex * 2 * np.pi * (-0.01 + 1j)
    poles[n_poles_real + 1::2] = pole_freqs_complex * 2 * np.pi * (-0.01 - 1j)

    return poles
