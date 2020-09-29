import numpy as np
from autograd import elementwise_grad as egrad
from abc import abstractmethod
from .gradient_descent import GradientDescentOptimizer


class ConjugateGradientsDescent(GradientDescentOptimizer):
    """
    Method of conjugate gradients

    Descent direction is the sum of gradient in current point
    and the weighted direction from the previous iteration
    """

    def __init__(self, x0, stop_criteria, step_optimizer, reset_iteration_number):
        super().__init__(x0, stop_criteria)
        self.__step_optimizer = step_optimizer
        self.__reset_iteration_number = reset_iteration_number

    @abstractmethod
    def _b_step(self, gradk, gradprev, sprev):
        pass

    def optimize(self, f):
        grad = egrad(f)

        xk = self._x0
        pk = np.zeros_like(xk)
        self._history = [xk, xk]

        iteration = 0
        while not self._stop_criteria.match(f, xk, self._history[-2]):
            grad_value = grad(xk)
            iteration += 1

            b = self._b_step(grad_value, grad(self._history[-2]), pk)

            if (self.__reset_iteration_number == iteration):  # reset pk for method convergence
                pk = 0
                iteration = 0

            pk = grad_value + b * pk

            a = self.__step_optimizer.optimize(
                lambda a: f(xk - a * pk))

            xk = xk - a * pk
            self._history.append(xk)

        self._history = self._history[1:]

        return xk


class FletcherReeves(ConjugateGradientsDescent):

    def _b_step(self, gradk, gradprev, sprev):
        numerator = np.linalg.norm(gradk)**2
        denominator = np.linalg.norm(gradprev)**2
        if np.linalg.norm(denominator) == 0:
            return 0
        else:
            return numerator / denominator


class HestenesStiefel(ConjugateGradientsDescent):

    def _b_step(self, gradk, gradprev, sprev):
        grad_dif = gradk - gradprev
        numerator = np.dot(gradk.T, grad_dif)
        denominator = -np.dot(sprev.T, grad_dif)
        if np.linalg.norm(denominator) == 0:
            return 0
        else:
            return numerator / denominator


class PolakRibier(ConjugateGradientsDescent):

    def _b_step(self, gradk, gradprev, sprev):
        grad_dif = gradk - gradprev
        numerator = np.dot(gradk.T, grad_dif)
        denominator = np.linalg.norm(gradprev)**2
        if np.linalg.norm(denominator) == 0:
            return 0
        else:
            return numerator / denominator


class DaiYuan(ConjugateGradientsDescent):

    def _b_step(self, gradk, gradprev, sprev):
        grad_dif = gradk - gradprev
        numerator = np.linalg.norm(gradk)**2
        denominator = -np.dot(sprev.T, grad_dif)
        if np.linalg.norm(denominator) == 0:
            return 0
        else:
            return numerator / denominator
