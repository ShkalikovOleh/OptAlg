from abc import ABC, abstractmethod
from optalg.optimizer import Optimizer
import numpy as np
from typing import Callable


class StepOptimizer(ABC):

    @abstractmethod
    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        pass
