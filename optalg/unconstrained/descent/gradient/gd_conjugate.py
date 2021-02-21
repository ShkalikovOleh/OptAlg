from typing import Callable
import numpy as np
from autograd import grad as agrad
from abc import abstractmethod
from ..descent_base import DescentOptimizerBase


class ConjugateGradientsDescent(DescentOptimizerBase):
    """
    Method of conjugate gradients

    Descent direction is the sum of gradient in current point
    and the weighted direction from the previous iteration
    """

    def __init__(self, stop_criterion, step_optimizer, renewal_step=None):
        super().__init__(stop_criterion, step_optimizer)
        if renewal_step is None:
            renewal_step = 10000
        self.__renewal_step = renewal_step

    @property
    def reset_iteration(self):
        return self.__renewal_step

    @reset_iteration.setter
    def reset_iteration(self, value):
        self.__renewal_step = value

    @abstractmethod
    def _b_step(self, gradk, gradprev, sprev):
        pass

    def _get_pk(self, f, xk):
        grad_value = self._grad(xk)
        pk = None

        if self._iteration_number % self.__renewal_step == 0:
            pk = grad_value
        else:
            pprev = self._phistory[-1]
            pk = grad_value + self._b_step(grad_value, self._pgrad, pprev) * pprev

        self._iteration_number += 1
        self._pgrad = grad_value

        return pk

    def optimize(self, f:Callable, x0: np.ndarray):
        self._grad = agrad(f)
        self._iteration_number = 0
        self._pgrad = np.zeros(shape=(x0.shape[0], 1))
        return super().optimize(f, x0)


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
