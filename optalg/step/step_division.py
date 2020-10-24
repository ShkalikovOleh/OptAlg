import numpy as np
from typing import Callable
from .step_optimizer import StepOptimizer


class StepDivision(StepOptimizer):
    """
    Chose initial step size and
    divide step by fixed value while function
    in the new point greater or equal function's
    value in the previous point.
    """

    def __init__(self, a, b):
        super().__init__()
        self.__a = a
        self.__b = b

    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        alphaK = self.__a
        xnew = xk - alphaK * pk

        while f(xk) <= f(xnew):
            alphaK = alphaK * self.__b
            xnew = xk - alphaK * pk

        return alphaK
