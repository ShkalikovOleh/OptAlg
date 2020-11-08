from abc import abstractmethod
import numpy as np
from ..optimizer import OptimizeResult, Optimizer


class DescentOptimizerBase(Optimizer):
    """
    Base class for method based descent to minimum
    """

    def __init__(self, x0: np.ndarray, stop_criterion, step_optimizer):
        super().__init__()
        self._x0 = x0.reshape(-1, 1)
        self._stop_criterion = stop_criterion
        self._step_optimizer = step_optimizer
        self._history = []
        self._phistory = []
        self._ahistory = []

    @property
    def x0(self):
        """
        Get starting point
        """
        return self._x0.reshape(self._x0.shape[0])

    @x0.setter
    def x0(self, value):
        """
        Set starting point
        """
        if value.size == self._x0.size:
            self._x0 = value.reshape(-1, 1)

    @abstractmethod
    def _get_pk(self, f, xk):
        """
        Get descent direction
        """
        pass

    def optimize(self, f):
        xk = self._x0

        self._history.append(xk)

        while not self._stop_criterion.match(f, xk, self._history[-1]):
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
