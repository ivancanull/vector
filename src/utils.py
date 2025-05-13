from ast import List
import numpy as np
from typing import Tuple

def float_p(string: str):
    """
    Convert a string to a float. If the conversion fails, return None.
    
    Parameters:
    - string: The string to be converted.
    
    Returns:
    - The converted float or None if conversion fails.
    """
    return float(string.replace('d', '.'))
    
def is_real(pole, atol=1e-12, rtol=1e-12):
    """
    Check if a given pole is real.
    
    Parameters:
    - pole: A complex number.
    
    Returns:
    - True if the pole is real, False otherwise.
    """
    return np.isclose(pole.imag, 0, atol=atol, rtol=rtol)

def frequency_to_s(frequency):
    """
    Convert a given frequency (in Hz) to the s-domain representation.
    
    Parameters:
    - frequency: A frequency value in Hz.
    
    Returns:
    - A complex number representing the s-domain equivalent.
    """
    omega = 2 * np.pi * frequency
    s = 1j * omega
    return s

def s_to_frequency(s):
    """
    Convert a given s-domain representation to the frequency (in Hz).
    
    Parameters:
    - s: A complex number in the s-domain.
    
    Returns:
    - The frequency value in Hz.
    """
    omega = np.imag(s)
    frequency = omega / (2 * np.pi)
    return frequency

def get_cindex(poles, atol=1e-12, rtol=1e-12):
    """
    Get the cindex array for the given poles. Assert that complex poles come in real values or conjugate pairs.
    """
    N = len(poles)
    cindex = np.zeros(N)
    i = 0
    while i < N:
        if not is_real(poles[i], atol=atol, rtol=rtol):
            assert np.allclose(np.conj(poles[i]), poles[i+1]) , ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]), '\n', 'All poles:', poles)
            cindex[i] = 1
            cindex[i+1] = 2
            i += 2
        else:
            i += 1
    return cindex

def sort_poles_residues(poles, residues=None, atol=1e-12, rtol=1e-12):
    """
    Sort the poles and residues in ascending order of real part.
    """
    # put the real poles first
    condition = is_real(poles, atol=atol, rtol=rtol)
    # sort the real poles
    real_poles = poles[condition].real # force the poles to be real
    complex_poles = poles[~condition]
    if residues is not None:
        real_residues = residues[condition]
        complex_residues = residues[~condition]
    
    # sort the complex poles by real part
    real_poles_idx = np.argsort(np.abs(real_poles))
    complex_poles_idx = np.argsort(np.abs(complex_poles))
    real_poles = real_poles[real_poles_idx]
    complex_poles = complex_poles[complex_poles_idx]

    if residues is not None:
        real_residues = real_residues[real_poles_idx]
        complex_residues = complex_residues[complex_poles_idx]
        return np.concatenate([real_poles, complex_poles]), np.concatenate([real_residues, complex_residues])
    else:
        return np.concatenate([real_poles, complex_poles])

def add_frequency_point_on_have(have_freq: np.ndarray, available_freq: np.ndarray, add_freq: float) -> Tuple[bool, np.ndarray]:
    """
    Add a frequency point on the available frequency points.
    
    Parameters:
    - have_freq: The frequency points that are already sampled.
    - available_freq: The available frequency points.
    - add_freq: The frequency point to be added.

    Returns:
    - Whether the frequency point is added or not.
    - The updated available frequency points.
    """
    # map the frequency point to the available frequency points
    mapped_add_freq = map_to_available_freq(add_freq, available_freq)
    # check if the frequency is already in have_freq
    if check_freq_sampled_already(have_freq, mapped_add_freq):
        return False, have_freq
    else:
        return True, np.sort(np.append(have_freq, mapped_add_freq))
    
def check_freq_sampled_already(have_freq: np.ndarray | List, freq: float, tol=1e-6):
    """
    Check if a frequency point is already sampled.
    
    Parameters:
    - have_freq: The frequency points that are already sampled.
    - freq: The frequency point to be checked.
    - tol: The tolerance to be used in the comparison.
    
    Returns:
    - True if the frequency point is already sampled, False otherwise.
    """
    if isinstance(have_freq, List):
        have_freq = np.array(have_freq)
    return np.any(np.isclose(have_freq, freq, atol=tol))

def map_to_available_freq(have_freq: np.ndarray | float, available_freq: np.ndarray) -> np.ndarray | float:
    """
    Map the available frequency points to the frequency points that are already sampled.
    
    Parameters:
    - have_freq: The frequency points that are already sampled.
    - available_freq: The available frequency points.
    
    Returns:
    - The available frequency points that are not already sampled.
    """
    mapped_freq = []
    if isinstance(have_freq, float):
        idx = np.argmin(np.abs(available_freq - have_freq))
        return available_freq[idx]
    elif isinstance(have_freq, np.ndarray):
        for i, freq in enumerate(have_freq):
            idx = np.argmin(np.abs(available_freq - freq))
            if not check_freq_sampled_already(mapped_freq, available_freq[idx]):
                mapped_freq.append(available_freq[idx])
        # sort the mapped frequency points
        mapped_freq = np.sort(np.array(mapped_freq))
        return mapped_freq
    else:
        raise ValueError("Invalid type for have_freq")
        
