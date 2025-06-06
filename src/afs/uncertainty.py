import numpy as np
from sympy import false

def uncertainty_aware_distribute(
    x: np.ndarray,
    u: np.ndarray,
    sample_num: int,
):
    """
    Uncertainty-aware distribution function for a given input and uncertainty.
    Args:
    - x (np.ndarray): Input data points.
    - u (np.ndarray): Uncertainty values corresponding to the input data points.
    - sample_num (int): Number of samples to generate from the distribution.

    Returns:
    - samples (np.ndarray): Samples drawn from the uncertainty-aware distribution.
    """

    pdf = u / np.sum(u)
    samples = np.random.choice(x, size=sample_num, replace=false, p=pdf)
    return samples.reshape(-1)  # Ensure samples are in the correct shape