from abc import abstractmethod
import numpy as np
from ..optimizer import OptimizerWithHistory


class DescentOptimizerBase(OptimizerWithHistory):
    """
    Base class for method based descent to minimum
    """

    def __init__(self, x0: np.ndarray, stop_criterion, step_optimizer):
        super().__init__()
        self._x0 = x0.reshape(-1, 1)
        self._stop_criterion = stop_criterion
        self._step_optimizer = step_optimizer
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

    @property
    def step_history(self):
        return np.array(self._ahistory)

    @property
    def direction_history(self):
        return np.array(self._phistory)[..., 0]

    @abstractmethod
    def _get_pk(self, f, xk):
        """
        Get descent direction
        """
        pass

    def optimize(self, f):
        xk = self._x0
        pk = np.zeros_like(xk)

        self.history_reset()
        self._history.append(xk)
        self._ahistory = []
        self._phistory = []

        while not self._stop_criterion.match(f, xk, self._get_prelast()):
            pk = self._get_pk(f, xk)
            a = self._step_optimizer.optimize(f, xk, pk)
            xk = xk - a * pk

            self._history.append(xk)
            self._ahistory.append(a)
            self._phistory.append(pk)

        return xk.reshape(xk.shape[0])
