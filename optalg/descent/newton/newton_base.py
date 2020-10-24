from abc import abstractmethod
import numpy as np

from autograd import elementwise_grad as egrad
from autograd import hessian

from typing import Callable
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

    def _get_pk(self, f: Callable, xk: np.ndarray, pprev: np.ndarray) -> np.ndarray:
        grad = self._grad(xk)
        hinv = self._get_inverse_h(xk)
        return np.dot(hinv, grad)

    def optimize(self, f: Callable) -> np.ndarray:
        self._grad = egrad(f)
        return super().optimize(f)
