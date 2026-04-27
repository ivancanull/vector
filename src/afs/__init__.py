from .Giulio import GiulioAFS
from .uncertainty import *


def SimonBayesianAFS(*args, **kwargs):
    # PyMC is an optional dependency for this adapter; delay importing it so
    # unrelated AFS and BNN workflows can import src without loading PyMC.
    from .Simon import SimonBayesianAFS as _SimonBayesianAFS

    return _SimonBayesianAFS(*args, **kwargs)
