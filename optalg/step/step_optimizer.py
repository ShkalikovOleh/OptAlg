import numpy as np
from typing import Callable
from abc import ABC, abstractmethod
from ..optimizer import Optimizer, OptimizeResult


class StepOptimizer(Optimizer):

    @abstractmethod
    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray) -> OptimizeResult:
        pass
