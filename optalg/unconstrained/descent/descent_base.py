from abc import abstractmethod
from typing import Callable, List
import numpy as np
from ...optimizer import OptimizeResult, Optimizer
from ...stop_criteria import StopCriterion
from ...line_search.line_searcher import LineSearcher
from ...collector import CollectorBase, reset_collectors, accept_collectors


class DescentOptimizerBase(Optimizer):
    """
    Base class for method based on the descent to minimum

    x_{k+1} = x_k - a_k * p_k
    """

    def __init__(self, stop_criterion: StopCriterion,
                 step_optimizer: LineSearcher,
                 x_collectors: List[CollectorBase] = None,
                 direction_collectors: List[CollectorBase] = None,
                 step_collectors: List[CollectorBase] = None):
        super().__init__()

        self._stop_criterion = stop_criterion
        self._step_optimizer = step_optimizer
        self._x_collectors = x_collectors
        self._direction_collectors = direction_collectors
        self._step_collectors = step_collectors

    @abstractmethod
    def _get_pk(self, f, xk):
        """
        Get descent direction
        """
        pass

    def optimize(self, f: Callable, x0: np.ndarray):

        xk = x0.reshape(-1, 1)

        n_iter = 0
        self._stop_criterion.accept(f, xk)

        reset_collectors(self._x_collectors)
        reset_collectors(self._direction_collectors)
        reset_collectors(self._step_collectors)

        accept_collectors(self._x_collectors, xk)

        while not self._stop_criterion.match():
            n_iter += 1

            pk = self._get_pk(f, xk)
            a = self._step_optimizer.optimize(f, xk, pk)
            xk = xk - a * pk

            self._stop_criterion.accept(f, xk)

            accept_collectors(self._x_collectors, xk)
            accept_collectors(self._step_collectors, a)
            accept_collectors(self._direction_collectors, pk)

        res = OptimizeResult(f=f, x=xk.reshape(-1),
                             n_iter=n_iter)
        return res
