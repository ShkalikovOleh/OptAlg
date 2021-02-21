import numpy as np
from typing import Callable
from autograd import elementwise_grad as egrad
from .line_searcher import LineSearcher


class BisectionWolfe(LineSearcher):
    """
    Bisection method that either computes a step size
    satisfying the weak Wolfe conditions or sends
    the function values to -inf
    """

    def __init__(self, a=1, c1=10**-4, c2=0.9) -> None:
        self.__a = a
        self.__c1 = c1
        self.__c2 = c2

    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        def gdk(x):
            return np.dot(pk.T, egrad(f)(x))

        ak = self.__a
        d = 0
        b = None

        while True:
            if f(xk - ak * pk) - f(xk) > - self.__c1 * ak * gdk(xk):
                b = ak
                ak = (d+b)/2
            elif gdk(xk - ak * pk) > self.__c2 * gdk(xk):
                d = ak
                if b is None:
                    ak = 2 * d
                else:
                    ak = (d+b)/2
            else:
                return ak
