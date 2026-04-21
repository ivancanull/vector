import re
import numpy as np
from .utils import get_cindex, sort_poles_residues

__all__ = ["fitted_error_estimate", "poles_residues_error_estimate"]

def fitted_error_estimate(Z, Z_fit, method="RMSE", **kwargs):
    """
    Calculate the error estimate between the fitted and original impedance. The default error calculation method is Root Mean Square Error (RMSE). 
    
    Parameters:
    - Z: The original impedance.
    - Z_fit: The fitted impedance.
    
    Returns:
    - The error estimate.
    """
    if method == "MAE": # Mean Absolute Error
        return np.mean(np.abs(Z - Z_fit))
    elif method == "MSE": # Mean Square Error
        return np.mean((np.abs(Z - Z_fit))**2)
    elif method == "RMSE":
        return np.sqrt(np.mean((np.abs(Z - Z_fit))**2))
    elif method == "NRMSE": # Normalized Root Mean Square Error
        norm = kwargs["norm"] if "norm" in kwargs else "range"
        rmse = np.sqrt(np.mean((np.abs(Z - Z_fit))**2))
        # Normalize RMSE based on the chosen method
        if norm == "range":
            return rmse / (np.max(np.abs(Z)) - np.min(np.abs(Z)))
        elif norm == "mean":
            return rmse / np.mean(np.abs(Z))
        elif norm == "std":
            return rmse / np.std(np.abs(Z))
        else:
            raise ValueError(f"Unknown normalization method: {norm}")
    elif method == "R-squared":
        SS_res = np.sum((np.abs(Z - Z_fit)) ** 2)
        SS_tot = np.sum((np.abs(Z) - np.mean(np.abs(Z))) ** 2)
        return 1 - (SS_res / SS_tot)
    elif method == "MReE": # Mean Relative Error
        return np.mean(np.abs((Z - Z_fit) / np.abs(Z)))
    elif method == "MxReE": # Maximum Relative Error
        return np.max(np.abs((Z - Z_fit) / np.abs(Z)))
    elif method == "NMReE": # Normalized Mean Relative Error   
        norm = kwargs["norm"] if "norm" in kwargs else "range"
        mre = np.mean(np.abs((Z - Z_fit) / np.abs(Z)))
        # Normalize MReE based on the chosen method
        if norm == "range":
            return mre / (np.max(np.abs(Z)) - np.min(np.abs(Z)))
        elif norm == "mean":
            return mre / np.mean(np.abs(Z))
        elif norm == "std":
            return mre / np.std(np.abs(Z))
        else:
            raise ValueError(f"Unknown normalization method: {norm}")
    elif method == "PSNR": # Peak Signal-to-Noise Ratio
        mse = np.mean((np.abs(Z - Z_fit))**2)
        max_val = np.max(np.abs(Z))
        if max_val == 0:
            return float('inf')
        else:
            return 20 * np.log10(max_val / np.sqrt(mse))
    else:
        raise ValueError("Invalid method.")

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
    poles, residues = sort_poles_residues(poles, residues)
    poles_fit, residues_fit = sort_poles_residues(poles_fit, residues_fit)

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
