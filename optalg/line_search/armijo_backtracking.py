import numpy as np
from typing import Callable
from autograd import grad as agrad
from .line_searcher import LineSearcher


class ArmijoBacktracking(LineSearcher):

    def __init__(self, a, b, c=10**-4) -> None:
        self.__a = a
        self.__b = b
        self.__c = c

    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        m = - self.__c * np.dot(pk.T, agrad(f)(xk))
        ak = self.__a

        while f(xk - ak * pk) - f(xk) > ak*m:
            ak = ak * self.__b

        return ak
