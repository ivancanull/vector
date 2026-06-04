from .objective_weighted import (
    ObjectiveWeightedAFSConfig,
    ObjectiveWeightedAFSResult,
    VFFitConfig,
    objective_band_weights,
    objective_vf_residual_db,
    objective_weighted_anchor_indices,
    reconstruct_sparams_vf,
    select_objective_weighted_afs_indices,
    uncertainty_profile_from_samples,
    write_touchstone_ri,
)
from .uncertainty import *


def GiulioAFS(*args, **kwargs):
    # Delay legacy imports so OWAFS/BNN code can import afs under PYTHONPATH=src
    # without pulling modules that use package-relative imports.
    from .Giulio import GiulioAFS as _GiulioAFS

    return _GiulioAFS(*args, **kwargs)


def SimonBayesianAFS(*args, **kwargs):
    # PyMC is an optional dependency for this adapter; delay importing it so
    # unrelated AFS and BNN workflows can import src without loading PyMC.
    from .Simon import SimonBayesianAFS as _SimonBayesianAFS

    return _SimonBayesianAFS(*args, **kwargs)


__all__ = [
    "GiulioAFS",
    "ObjectiveWeightedAFSConfig",
    "ObjectiveWeightedAFSResult",
    "SimonBayesianAFS",
    "VFFitConfig",
    "objective_band_weights",
    "objective_vf_residual_db",
    "objective_weighted_anchor_indices",
    "reconstruct_sparams_vf",
    "select_objective_weighted_afs_indices",
    "uncertainty_profile_from_samples",
    "write_touchstone_ri",
]
