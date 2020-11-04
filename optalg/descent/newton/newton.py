import numpy as np
from typing import Callable
from autograd import hessian
from .newton_base import NewtonBase


class Newton(NewtonBase):

    def __init__(self, x0, stop_criterion, step_optimizer) -> None:
        super().__init__(x0, stop_criterion, step_optimizer)

    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        h = self._hessian(xk)
        h = h.reshape((h.shape[1], h.shape[1]))
        return np.linalg.inv(h)

    def optimize(self, f: Callable) -> np.ndarray:
        self._hessian = hessian(f)
        return super().optimize(f)
