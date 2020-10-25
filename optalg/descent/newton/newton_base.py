import numpy as np
from typing import Callable
from abc import abstractmethod
from autograd import elementwise_grad as egrad
from ..descent_base import DescentOptimizerBase


class NewtonBase(DescentOptimizerBase):
    """
    Base class for newton and quasinewton second
    order optimization methods, which use hessian and gradient
    for descent direction calculation.
    """

    def __init__(self, x0, stop_criterion, step_optimizer) -> None:
        super().__init__(x0, stop_criterion, step_optimizer)

    @abstractmethod
    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        pass

    def _get_pk(self, f: Callable, xk: np.ndarray) -> np.ndarray:
        grad = self._grad(xk)
        hinv = self._get_inverse_h(xk)
        self._pgrad = grad  # caching for quasi newton
        self._phinv = hinv  # caching for quasi newton
        return np.dot(hinv, grad)

    def optimize(self, f: Callable) -> np.ndarray:
        self._grad = egrad(f)
        self._pgrad = np.zeros_like(self._grad)
        self._phinv = np.zeros(shape=(self._x0.shape[0], self._x0.shape[0]))
        return super().optimize(f)
