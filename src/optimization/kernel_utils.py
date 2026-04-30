from __future__ import annotations

import numpy as np

from .datasets import PassiveDataset


def build_geometry_matrix(dataset: PassiveDataset, indices: list[int]) -> np.ndarray:
    """Create a dense geometry matrix for optimizer kernels.

    This utility is shared by replay now and reserved for OptimizerBoTorch migration
    so both optimizers use identical candidate preprocessing.
    """
    if not indices:
        return np.empty((0, 0), dtype=float)
    return np.asarray([dataset.get_geometry(i) for i in indices], dtype=float)
