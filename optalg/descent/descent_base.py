from abc import abstractmethod
import numpy as np
from ..optimizer import OptimizerWithHistory


class DescentOptimizerBase(OptimizerWithHistory):
    """
    Base class for method based on descent to minimum
    -pk - descent direction
    a - learning rate(>0)
    """

    def __init__(self, x0, stop_criterion, step_optimizer):
        super().__init__()
        self._x0 = x0
        self._stop_criterion = stop_criterion
        self._step_optimizer = step_optimizer

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        if value.shape == self._x0.shape:
            self._x0 = value

    @abstractmethod
    def _get_pk(self, f, xk, pprev):
        pass

    def optimize(self, f):
        xk = self._x0

        self.history_reset()
        self.append_history(xk)
        pk = np.zeros_like(xk)

        while not self._stop_criterion.match(f, xk, self._get_prelast()):
            pk = self._get_pk(f, xk, pk)
            a = self._step_optimizer.optimize(f, xk, pk)
            xk = xk - a * pk
            self._history.append(xk)

        return xk
