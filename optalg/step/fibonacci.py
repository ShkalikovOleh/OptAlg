from optalg.optimizer import OptimizeResult
import numpy as np
from typing import Callable
from .step_optimizer import StepOptimizer
from ..optimizer import OptimizeResult


class Fibonacci(StepOptimizer):
    """
    Find argmin of one-dimensional UNIMODAL function in bounds

    Drawback: Algorithm is very sensitive to float errors
    """

    def __init__(self, bounds, epsilon):
        """
        Init iterative search object

        Args:
            bounds (array-like): bounds of search. Must contain 2 values.
                                Second value must be larger than first.
            epsilon (float): calculation precision
        """
        super().__init__()
        self.__a, self.__b = bounds
        self.__epsilon = epsilon

    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        def g(a):
            return f(xk - a*pk)

        diam = self.__b - self.__a
        F1, F2, F3 = 1, 1, 2
        j = 1
        while not (F2 < diam/self.__epsilon <= F3):
            F1, F2, F3 = F2, F3, F2+F3
            j = j+1
        m = j

        k = diam * F1/F3
        y = self.__a + k
        z = self.__b - k

        if g(y) <= g(z):
            a = self.__a
            b = z
        else:
            a = y
            b = self.__b

        k = 1
        while k < m-1:
            if g(y) <= g(z):
                z = y
                y = a + b - y
            else:
                y = z
                z = a + b - z

            if g(y) <= g(z):
                b = z
            else:
                a = y
            k = k+1
        X = (a+b)/2

        return OptimizeResult(x=X)


class ModFibonacci(StepOptimizer):
    """
    Find argmin of one-dimensional UNIMODAL function in bounds
    Modified Fibonacci. More float-error-prone.
    """

    def __init__(self, bounds, epsilon):
        """
        Init iterative search object

        Args:
            bounds (array-like): bounds of search. Must contain 2 values.
                                Second value must be larger than first.
            epsilon (float): calculation precision
        """
        super().__init__()
        self.__a, self.__b = bounds
        self.__epsilon = epsilon

    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        def g(a):
            return f(xk - a*pk)

        diam = self.__b - self.__a
        F = [1, 1]
        j = 1
        F.append(F[j]+F[j-1])
        while not (F[j] < diam/self.__epsilon <= F[j+1]):
            F.append(F[j+1]+F[j])
            j = j+1
        m = j

        y = self.__a + diam * F[m-1]/F[m+1]
        z = self.__a + diam * F[m]/F[m+1]

        if g(y) <= g(z):
            a = self.__a
            b = z
        else:
            a = y
            b = self.__b

        k = 1
        while k < m-1:
            if g(y) <= g(z):
                z = y
                y = a + diam * F[m-k-1]/F[m+1]
            else:
                y = z
                z = a + diam * F[m-k]/F[m+1]

            if g(y) <= g(z):
                b = z
            else:
                a = y
            k = k+1
        X = (a+b)/2

        return OptimizeResult(x=X)
