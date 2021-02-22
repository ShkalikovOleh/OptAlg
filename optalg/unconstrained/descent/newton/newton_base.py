import numpy as np
from typing import Callable, List
from abc import abstractmethod
from autograd import grad as agrad
from ..descent_base import DescentOptimizerBase
from ....line_search.line_searcher import LineSearcher
from ....stop_criteria import StopCriterion
from ....collector import CollectorBase, SaveLastCollector, accept_collectors, reset_collectors


class NewtonBase(DescentOptimizerBase):
    """
    Base class for newton and quasinewton second
    order optimization methods, which use hessian and gradient
    for descent direction calculation.
    """

    def __init__(self, stop_criterion: StopCriterion,
                 step_optimizer: LineSearcher,
                 x_collectors: List[CollectorBase] = None,
                 direction_collectors: List[CollectorBase] = None,
                 step_collectors: List[CollectorBase] = None,
                 hessian_collectors: List[CollectorBase] = None) -> None:

        super().__init__(stop_criterion, step_optimizer,
                         x_collectors, direction_collectors, step_collectors)

        self._last_hessian_collector = SaveLastCollector()
        if hessian_collectors is None:
            hessian_collectors = [self._last_hessian_collector]
        else:
            hessian_collectors.append(self._last_hessian_collector)
        self._hessian_collectors = hessian_collectors

    @abstractmethod
    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        pass

    def _get_pk(self, f: Callable, xk: np.ndarray) -> np.ndarray:
        grad = self._grad(xk)
        hinv = self._get_inverse_h(xk)

        self._pgrad = grad  # caching for quasi newton
        accept_collectors(self._hessian_collectors, hinv)

        return hinv @ grad

    def optimize(self, f: Callable, x0: np.ndarray) -> np.ndarray:
        self._grad = agrad(f)
        self._pgrad = np.zeros_like(self._grad)
        reset_collectors(self._hessian_collectors)

        return super().optimize(f, x0)


class QuasiNewtonBase(NewtonBase):

    def __init__(self, stop_criterion: StopCriterion,
                 step_optimizer: LineSearcher,
                 x_collectors: List[CollectorBase] = None,
                 direction_collectors: List[CollectorBase] = None,
                 step_collectors: List[CollectorBase] = None,
                 hessian_collectors: List[CollectorBase] = None) -> None:

        self._last_step_collector = SaveLastCollector()
        if step_collectors is None:
            step_collectors = [self._last_step_collector]
        else:
            step_collectors.append(self._last_step_collector)

        self._last_direction_collector = SaveLastCollector()
        if direction_collectors is None:
            direction_collectors = [self._last_direction_collector]
        else:
            direction_collectors.append(self._last_direction_collector)

        super().__init__(stop_criterion, step_optimizer,
                         x_collectors, direction_collectors,
                         step_collectors, hessian_collectors)
