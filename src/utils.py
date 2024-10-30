import numpy as np

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

def get_cindex(poles):
    """
    Get the cindex array for the given poles. Assert that complex poles come in real values or conjugate pairs.
    """
    N = len(poles)
    cindex = np.zeros(N)
    
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert np.conj(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2
    return cindex