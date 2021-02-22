import numpy as np
from typing import List
from .newton_base import QuasiNewtonBase
from ....stop_criteria import StopCriterion
from ....line_search.line_searcher import LineSearcher
from ....collector import CollectorBase


class BFGS(QuasiNewtonBase):

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
        I = np.eye(xk.shape[0])

        hk = self._last_hessian_collector.get_last()
        if hk is None:
            return I

        yk = self._grad(xk) - self._pgrad
        pk = self._last_step_collector.get_last() * self._last_direction_collector.get_last()

        yTp = np.dot(yk.T, pk)
        ypT = np.dot(yk, pk.T)
        pyT = np.dot(pk, yk.T)

        a = (I - pyT/yTp) @ hk @ (I - ypT/yTp)
        b = pyT / yTp

        return a + b
