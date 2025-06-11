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

def uniform_distribute(
    x: np.ndarray,
    sample_num: int,
):
    """
    Uniform distribution function for a given input.
    Args:
    - x (np.ndarray): Input data points.
    - sample_num (int): Number of samples to generate from the distribution.

    Returns:
    - samples (np.ndarray): Samples drawn from the uniform distribution.
    """

    indices = np.linspace(0, len(x)-1, num=sample_num, dtype=np.float32)
    indices = np.round(indices).astype(np.int32)
    indices = np.clip(indices, 0, len(x)-1)
    return x[indices].reshape(-1)  # Ensure samples are in the correct shape


