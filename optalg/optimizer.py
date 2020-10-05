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
        """
        Returns history by steps(with start point(s))
        If each step defines by one point, return matrix (n x m)-
        where n - variables count, m - steps count + 1
        Otherwise returns 'tensor' (m x l x n) where l - points count of each step
        """

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
