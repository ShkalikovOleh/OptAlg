from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):

    @abstractmethod
    def optimize(self, f):
        """
        Optimize function

        Args:
            f (callable): function for optimize

        Example:
            def f(x):
                return x**2 + 4

            optimizer = SomeOptimizer(parameters ...)
            xmin = optimizer.optimize(f)
        """
        pass


class OptimizerWithHistory(Optimizer):

    def __init__(self):
        super().__init__()
        self._history = []

    def get_last_history(self):
        arr = np.array(self._history)[..., 0]
        if arr.size > 0:
            if arr.ndim <= 2:
                return arr.T
            else:
                return arr
        else:
            return arr

    def history_reset(self):
        self._history = []
