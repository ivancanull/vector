from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

__all__ = ["PolesInitializer", "BasicPolesInitializer"]
class PolesInitializer(ABC):
    """
    Abstract class for poles initializer.
    """
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self):
        pass


class BasicPolesInitializer(PolesInitializer):
    """
    Basic poles initializer for Vector Fitting algorithm. The poles are initialized based on the frequency range with given sampling method.

    Parameters:

    - freqs_range: A tuple representing the frequency range (fmin, fmax).
    - n_poles_real: Number of real poles.
    - n_poles_complex: Number of complex poles conjugate pairs.

    Reference: https://github.com/scikit-rf/scikit-rf/blob/master/skrf/vectorFitting.py

    
    """
    def __init__(self, freqs_range: Tuple, n_poles_real: int, n_poles_complex: int):
        self.fmin = freqs_range[0]
        self.fmax = freqs_range[1]

        if self.fmin == 0.0:
            self.fmin = self.fmax / 1e3
        
        self._n_poles_real = n_poles_real    
        self._n_poles_complex = n_poles_complex

    def set_poles_num(self, n_poles_real, n_poles_complex):
        self._n_poles_real = n_poles_real
        self._n_poles_complex = n_poles_complex

    def initialize(self, scale="log"):
        if scale == "log":
            pole_freqs_real = np.geomspace(self.fmin, self.fmax, self._n_poles_real)
            pole_freqs_complex = np.geomspace(self.fmin, self.fmax, self._n_poles_complex)
        elif scale == "linear":
            pole_freqs_real = np.linspace(self.fmin, self.fmax, self._n_poles_real)
            pole_freqs_complex = np.linspace(self.fmin, self.fmax, self._n_poles_complex)
        
        else:
            raise ValueError("Invalid scale. Choose 'log' or 'linear'.")

        poles = np.zeros(self._n_poles_real + 2 * self._n_poles_complex, dtype=complex)
        poles[:self._n_poles_real] = -pole_freqs_real * 2 * np.pi
        poles[self._n_poles_real::2] = pole_freqs_complex * 2 * np.pi * (-0.01 + 1j)
        poles[self._n_poles_real + 1::2] = pole_freqs_complex * 2 * np.pi * (-0.01 - 1j)

        return poles

if __name__ == "__main__":
    initializer = BasicPolesInitializer(freqs_range=(1, 1e6), n_poles_real=1, n_poles_complex=4)
    poles = initializer.initialize(scale="linear")
    print(poles) 


        