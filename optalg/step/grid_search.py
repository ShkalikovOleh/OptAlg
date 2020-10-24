import numpy as np
from typing import Callable
from .step_optimizer import StepOptimizer


class GridSearch(StepOptimizer):
    """
    Search argmin step of the function for direction
    by iterating over n values in bounds
    """

    def __init__(self, bounds, n):
        """
        Init grid search

        Args:
            bounds (array-like): bounds of search. Must contains 2 value.
                                Second value must be larger than first.
            n (int): number of iteration
        """
        super().__init__()
        self.__bounds = bounds
        self.__n = n

    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        a = np.linspace(self.__bounds[0], self.__bounds[1], self.__n)
        y = f(xk - a*pk)
        return a[np.argmin(y)]
