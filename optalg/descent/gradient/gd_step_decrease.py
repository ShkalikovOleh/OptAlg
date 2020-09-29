import numpy as np
from autograd import elementwise_grad as egrad
from .gradient_descent import GradientDescentOptimizer


class GradientDescentStepDecrease(GradientDescentOptimizer):

    def __init__(self, x0, alpha, beta, stop_criteria):
        super().__init__(x0, stop_criteria)
        self.__alpha = alpha
        self.__beta = beta

    def optimize(self, f):
        grad = egrad(f)

        xk = self._x0

        self._history = [xk, xk]

        while not self._stop_criteria.match(f, xk, self._history[-2]):
            grad_value = grad(xk)

            alphaK = self.__alpha
            xnew = xk - alphaK * grad_value

            while f(xk) <= f(xnew):
                alphaK = alphaK * self.__beta
                xnew = xk - alphaK * grad_value

            xk = xnew
            self._history.append(xk)

        self._history = self._history[1:]

        return xk
