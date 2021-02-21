import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class LineSearcher(ABC):

    @abstractmethod
    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        pass
