import numpy as np
from abc import abstractmethod
from autograd import grad as agrad
from typing import Callable, List
from ..descent_base import DescentOptimizerBase
from ....stop_criteria import StopCriterion
from ....line_search.line_searcher import LineSearcher
from ....collector import SaveLastCollector, CollectorBase


class ConjugateGradientsDescent(DescentOptimizerBase):
    """
    Method of conjugate gradients

    Descent direction is the sum of gradient in current point
    and the weighted direction from the previous iteration
    """

    def __init__(self, stop_criterion: StopCriterion,
                 step_optimizer: LineSearcher,
                 renewal_step: int = 1000,
                 x_collectors: List[CollectorBase] = None,
                 direction_collectors: List[CollectorBase] = None,
                 step_collectors: List[CollectorBase] = None):

        self._dir_collector = SaveLastCollector()
        if direction_collectors is not None:
            direction_collectors.append(self._dir_collector)
        else:
            direction_collectors = [self._dir_collector]

        super().__init__(stop_criterion, step_optimizer,
                         x_collectors, direction_collectors, step_collectors)
        self.__renewal_step = renewal_step

    @abstractmethod
    def _b_step(self, gradk, gradprev, sprev):
        pass

    def _get_pk(self, f, xk):
        grad_value = self._grad(xk)
        pk = None

        if self._iteration_number % self.__renewal_step == 0:
            pk = grad_value
        else:
            pprev = self._dir_collector.get_last()
            pk = grad_value + \
                self._b_step(grad_value, self._pgrad, pprev) * pprev

        self._iteration_number += 1
        self._pgrad = grad_value

        return pk

    def optimize(self, f: Callable, x0: np.ndarray):
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
