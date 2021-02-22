import numpy as np
from typing import List
from .newton_base import QuasiNewtonBase
from ....stop_criteria import StopCriterion
from ....line_search.line_searcher import LineSearcher
from ....collector import CollectorBase


class Broyden(QuasiNewtonBase):

    def __init__(self, stop_criterion: StopCriterion,
                 step_optimizer: LineSearcher,
                 x_collectors: List[CollectorBase] = None,
                 direction_collectors: List[CollectorBase] = None,
                 step_collectors: List[CollectorBase] = None,
                 hessian_collectors: List[CollectorBase] = None) -> None:
        super().__init__(stop_criterion, step_optimizer,
                         x_collectors, direction_collectors,
                         step_collectors, hessian_collectors)

    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        hk = self._last_hessian_collector.get_last()
        if hk is None:
            return np.eye(xk.shape[0])

        gk = self._grad(xk) - self._pgrad
        pk = self._last_step_collector.get_last() * self._last_direction_collector.get_last()

        numerator = (pk - hk @ gk) @ pk.T @ hk
        denominator = pk.T @ hk @ gk

        if denominator == 0:
            return np.eye(xk.shape[0])
        return hk + numerator / denominator
