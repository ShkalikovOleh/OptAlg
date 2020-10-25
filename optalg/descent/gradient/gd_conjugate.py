import numpy as np
from autograd import elementwise_grad as egrad
from abc import abstractmethod
from ..descent_base import DescentOptimizerBase


class ConjugateGradientsDescent(DescentOptimizerBase):
    """
    Method of conjugate gradients

    Descent direction is the sum of gradient in current point
    and the weighted direction from the previous iteration
    """

    def __init__(self, x0, stop_criterion, step_optimizer, reset_iteration_number=None):
        super().__init__(x0, stop_criterion, step_optimizer)
        if reset_iteration_number is None:
            reset_iteration_number = 10000
        self.__reset_iteration_number = reset_iteration_number

    @property
    def reset_iteration(self):
        return self.__reset_iteration_number

    @reset_iteration.setter
    def reset_iteration(self, value):
        self.__reset_iteration_number = value

    @abstractmethod
    def _b_step(self, gradk, gradprev, sprev):
        pass

    def _get_pk(self, f, xk):
        grad_value = self._grad(xk)
        pk = None

        self._iteration_number += 1
        if self.__reset_iteration_number <= self._iteration_number or len(self._phistory) == 0:
            self._iteration_number = 0
            pk = grad_value
        else:
            pprev = self._phistory[-1]
            pk = grad_value + self._b_step(grad_value, self._pgrad, pprev) * pprev

        self._pgrad = grad_value

        return pk

    def optimize(self, f):
        self._grad = egrad(f)
        self._iteration_number = 0
        self._pgrad = np.zeros_like(self._x0)
        return super().optimize(f)


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
