import numpy as np
from autograd import hessian
from typing import Callable
from ..descent_base import FastestDescentBase
from .newton_base import NewtonBase


class SimpleNewtonBase(NewtonBase):
    def __init__(self, x0, stop_criterion, **kwargs) -> None:
        super().__init__(x0, stop_criterion)

    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        h = self._hessian(xk)
        h = h.reshape((h.shape[1], h.shape[1]))
        return np.linalg.inv(h)

    def optimize(self, f: Callable):
        self._hessian = hessian(f)
        return super().optimize(f)


class Newton(SimpleNewtonBase):
    def __init__(self, x0, stop_criterion, step_opt) -> None:
        super().__init__(x0=x0, stop_criterion=stop_criterion, step_opt=step_opt)
        self._step_opt = step_opt

    def _get_a(self, f, xk, pk):
        return self._step_opt.optimize(lambda a: f(xk - a*pk))
