from abc import abstractmethod
from typing import Callable
import numpy as np
from ..optimizer import OptimizeResult, Optimizer


class DescentOptimizerBase(Optimizer):
    """
    Base class for method based descent to minimum
    """

    def __init__(self, stop_criterion, step_optimizer):
        super().__init__()

        self._stop_criterion = stop_criterion
        self._step_optimizer = step_optimizer
        self._history = []
        self._phistory = []
        self._ahistory = []

    @abstractmethod
    def _get_pk(self, f, xk):
        """
        Get descent direction
        """
        pass

    def optimize(self, f: Callable, x0: np.ndarray):
        xk = x0.reshape(-1, 1)

        self._history.append(xk)

        while not self._stop_criterion.match(f, self._history):
            pk = self._get_pk(f, xk)
            a = self._step_optimizer.optimize(f, xk, pk).x
            xk = xk - a * pk

            self._history.append(xk)
            self._ahistory.append(a)
            self._phistory.append(pk)

        xhist = np.array(self._history).reshape(
            (len(self._history), xk.shape[0]))

        res = OptimizeResult(f=f, x=xk.reshape(-1),
                             x_history=xhist,
                             step_history=np.array(self._ahistory),
                             direction_history=np.array(self._phistory)[..., 0])

        self._history.clear()
        self._ahistory.clear()
        self._phistory.clear()

        return res
