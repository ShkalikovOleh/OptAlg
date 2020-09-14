import numpy as np
from ..optimizer import Optimizer


class SimpleSearch(Optimizer):
    """
    Search argmin of one-value function from n values in bounds
    """

    def __init__(self, bounds, n):
        """
        Init iterative search object

        Args:
            bounds (array-like): bounds of search. Must contains 2 value.
                                Second value must be larger than first.
            n (int): number of iteration
        """
        super().__init__()
        self.__bounds = bounds
        self.__n = n

    def optimize(self, f):
        x = np.linspace(self.__bounds[0], self.__bounds[1], self.__n)
        y = f(x)
        return x[np.argmin(y)]
