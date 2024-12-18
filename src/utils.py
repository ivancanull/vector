from calendar import c
import numpy as np

def is_real(pole):
    """
    Check if a given pole is real.
    
    Parameters:
    - pole: A complex number.
    
    Returns:
    - True if the pole is real, False otherwise.
    """
    return np.isclose(pole.imag, 0)

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

def get_cindex(poles, tol=1e-12):
    """
    Get the cindex array for the given poles. Assert that complex poles come in real values or conjugate pairs.
    """
    N = len(poles)
    cindex = np.zeros(N)
    i = 0
    while i < N:
        if np.abs(poles[i].imag / poles[i].real) > tol:
            assert np.allclose(np.conj(poles[i]), poles[i+1]) , ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
            cindex[i] = 1
            cindex[i+1] = 2
            i += 2
        else:
            i += 1
    return cindex