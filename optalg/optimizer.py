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
        arr = np.array(self._history)
        if arr.size > 0:
            return arr[..., 0].T
        else:
            return arr

    def append_history(self, xs):
        self._history.append(xs)

    def history_reset(self):
        self._history = []

    def _get_prelast(self):
        """
        Get previous of the last elements in the history
        If the history contains only 1 element return last
        """
        if len(self._history) == 0:
            return None
        elif len(self._history) == 1:
            return self._history[0]
        else:
            return self._history[-2]
