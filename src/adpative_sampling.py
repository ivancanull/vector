from .poles_init import PolesInitializer
from .poles_init import *

from typing import Tuple, Callable
from abc import ABC, abstractmethod

class SamplingStrategy(ABC):
    """
    Abstract class for sampling strategy.
    """
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

class AdaptiveSampling(SamplingStrategy):
    """
    Adaptive Sampling algorithm for fast and accurate S-parameter modeling of general planar structures.

    Parameters:

    Reference: Dhaene, Tom, et al. "Adaptive frequency sampling algorithm for fast and accurate S-parameter modeling of general planar structures." Proceedings of 1995 IEEE MTT-S International Microwave Symposium. IEEE, 1995.

    
    """
    def __init__(self, 
                 freqs_range: Tuple,
                 poles_initializor: PolesInitializer,):
        
        self.fmin = freqs_range[0]
        self.fmax = freqs_range[1]
        self.poles_initializor = poles_initializor

    