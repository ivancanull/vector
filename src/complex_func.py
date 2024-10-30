import numpy as np

__all__ = ["ComplexFunction"]

class ComplexFunction:
    def __init__(self, poles, residues, d=0, h=0):
        """
        Initialize the complex function with poles, residues, and coefficients d and h.
        
        Parameters:
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
        
        Parameters:
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
        
        Parameters:
        - s: A complex number or a numpy array of complex numbers where the function is to be evaluated.

        Returns:
        - The value of the function at s or an array of values if s is an array
        """
        return self.evaluate(s)
    
    