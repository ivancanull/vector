import re
import numpy as np
from .utils import get_cindex

__all__ = ["fitted_error_estimate", "poles_residues_error_estimate", "poles_sorting"]

def fitted_error_estimate(Z, Z_fit, method="RMSE"):
    """
    Calculate the error estimate between the fitted and original impedance. The error calculation method is Root Mean Square Error (RMSE). 
    
    Parameters:
    - Z: The original impedance.
    - Z_fit: The fitted impedance.
    
    Returns:
    - The error estimate.
    """
    if method == "RMSE":
        return np.sqrt(np.mean((np.abs(Z - Z_fit))**2))
    else:
        raise ValueError("Invalid method. Choose from 'RMSE'.")

def poles_residues_error_estimate(poles, residues, d, h, poles_fit, residues_fit, d_fit, h_fit, method="Error"):
    """
    Calculate the error estimate between the fitted and original poles and residues. The error calculation method supports:
    - Error: Direct error calculation.
    - Sort: Sort the poles and residues in ascending order of real part.
    
    Parameters:
    - poles: The original poles.
    - residues: The original residues.
    - d: The original d.
    - h: The original h.
    - poles_fit: The fitted poles.
    - residues_fit: The fitted residues.
    - d_fit: The fitted d.
    - h_fit: The fitted h.
    
    Returns:
    - The error estimate.
    """

    # sort the poles and residues in ascending order of real part
    poles, residues = poles_sorting(poles, residues)
    poles_fit, residues_fit = poles_sorting(poles_fit, residues_fit)

    if method == "Error":
        poles_error = poles - poles_fit
        residues_error = residues - residues_fit
        d_error = d - d_fit
        h_error = h - h_fit
        return poles_error, residues_error, d_error, h_error
    elif method == "Sort":
        return poles, residues, poles_fit, residues_fit
    else:
        raise ValueError("Invalid method. Choose from 'Error', 'Sort'.")
    
def poles_sorting(poles, residues):
    """
    Sort the poles in ascending order of real part of the poles. Since the poles are real or conjugate pairs, the sorting of conjugate pairs is done with positive real part first.
    
    Parameters:
    - poles: The poles to be sorted.
    - residues: The residues corresponding to the poles.
    
    Returns:
    - The sorted poles, residues, d, and h.
    """
    poles = np.array(poles)
    residues = np.array(residues)
    N = len(poles)
    cindex = get_cindex(poles)
    
    # argsort the poles
    sorted_indices = np.argsort(poles, kind='mergesort')

    poles = poles[sorted_indices]
    residues = residues[sorted_indices]
    cindex = cindex[sorted_indices]
    
    return poles, residues