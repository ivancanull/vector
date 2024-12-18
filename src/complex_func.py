from abc import ABC, abstractmethod
from typing import Sequence
from .utils import s_to_frequency
from .touchstone import TorchstoneParser
import numpy as np
from typing import Tuple
__all__ = ["ComplexFunction", "FunctionFromTouchstone"]

class FunctionAbstract(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def evaluate(self, s):
        pass

    @abstractmethod
    def __call__(self, s):
        pass

class ComplexFunction(FunctionAbstract):
    def __init__(self, poles, residues, d=0, h=0):
        """
        Initialize the complex function with poles, residues, and coefficients d and h.
        
        Args:
        - poles: A list or numpy array of complex numbers representing the poles.
        - residues: A list or numpy array of complex numbers representing the residues associated with the poles.
        - d: A complex constant, default is 0.
        - h: A complex constant representing the polynomial coefficient of s in the numerator, default is 0.
        """
        if len(poles) != len(residues):
            raise ValueError("Poles and residues must be of the same length.")
        
        self.poles = np.array(poles)
        self.residues = np.array(residues)
        self.d = d
        self.h = h
    
    def evaluate(self, s):
        """
        Evaluate the complex function at a given point or array of points s.
        
        Args:
        - s: A complex number or a numpy array of complex numbers where the function is to be evaluated.
        
        Returns:
        - The value of the function at s or an array of values if s is an array
        """
        s = np.array(s, dtype=complex)  # Ensure s is a numpy array
        value = self.d + self.h * s  # Initialize with d + hs (handling the polynomial term)

        # Compute the sum of residues/(s - poles)
        for pole, residue in zip(self.poles, self.residues):
            value += residue / (s - pole)
        
        return value
    
    def __call__(self, s):
        """
        Makes the instance callable, so you can use the instance like a function.
        
        Args:
        - s: A complex number or a numpy array of complex numbers where the function is to be evaluated.

        Returns:
        - The value of the function at s or an array of values if s is an array
        """
        return self.evaluate(s)
    
class FunctionFromTouchstone(FunctionAbstract):
    def __init__(self, filename, port_num, port_combinations: Tuple = None):
        """
        Initialize the function from a touchstone file.
        
        Args:
        - filename: The name of the touchstone file to be parsed.
        - port_num: The number of ports in the touchstone file.
        - port_combinations: A slice object representing the port combinations to be used in the function. Default is None.
        """
        self.filename = filename
        self.parser = TorchstoneParser(port_num=port_num)
        self.data = self.parser.parse(filename)
        self.port_combinations = port_combinations

    def evaluate(self, s):
        """
        Evaluate the function at a given point or array of points s.
        
        Args:
        - s: A complex number or a numpy array of complex numbers where the function is to be evaluated.
        
        Returns:
        - The value of the function at s or an array of values if s is an array
        """


        # find the closest frequency in the touchstone data
        freq = np.array(self.data['freq'])
        idx = np.argmin(np.abs(freq - s_to_frequency(s)))
        values = np.zeros((self.parser.port_num, self.parser.port_num), dtype=complex)
        for i in range(self.parser.port_num):
            for j in range(self.parser.port_num):
                if self.parser.options['format'] == 'db':
                    # value = self.data[f'{i+1}{j+1}_1'].iloc[idx] * np.exp(1j * np.radians(self.data[f'{i+1}{j+1}_2'].iloc[idx]))
                    value = 10 ** (self.data[f'{i+1}{j+1}_1'].iloc[idx] / 20) * np.exp(1j * np.radians(self.data[f'{i+1}{j+1}_2'].iloc[idx]))
                elif self.parser.options['format'] == 'ma':
                    value = self.data[f'{i+1}{j+1}_1'][idx] * np.exp(1j * np.radians(self.data[f'{i+1}{j+1}_2'][idx]))
                elif self.parser.options['format'] == 'ri':
                    value = self.data[f'{i+1}{j+1}_1'][idx] + 1j * self.data[f'{i+1}{j+1}_2'][idx]
                values[i, j] = value
        if self.port_combinations is None:
            return values
        else:
            return values[self.port_combinations[0], self.port_combinations[1]]
    def __call__(self, s):
        """
        Makes the instance callable, so you can use the instance like a function.

        Args:
        - s: A complex number or a numpy array of complex numbers where the function is to be evaluated.
        """       
        if isinstance(s, complex):
            return self.evaluate(s) 
        elif isinstance(s, np.ndarray):
            return np.array([self.evaluate(si) for si in s])
        else:
            raise ValueError("s must be a complex number or a numpy array of complex numbers.")
        
            