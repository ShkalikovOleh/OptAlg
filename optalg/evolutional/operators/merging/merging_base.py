import numpy as np
from typing import Callable
from abc import ABC, abstractmethod
from ..decoder import Decoder


class MergingOperator(ABC):
    """
    Merge a parents and children in new population
    """

    @abstractmethod
    def __call__(self, parents: np.ndarray,
                 children: np.ndarray, f: Callable, decoder: Decoder) -> np.ndarray:
        pass
