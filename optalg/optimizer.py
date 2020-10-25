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

    @property
    def history(self):
        """
        Returns history by steps(with start point(s))
        History has shape (m x l x n) where
        m - steps count + 1,
        l - points count of each step,
        n - variables count
        """


        arr = np.array(self._history)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]

        if arr.ndim == 2:
            arr = arr.reshape(arr.shape[0], 1, arr.shape[1])

        return arr

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
