import numpy as np
from typing import Callable
from autograd import grad as agrad
from ..descent_base import DescentOptimizerBase


class GradientDescent(DescentOptimizerBase):

    def __init__(self, stop_criterion, step_optimizer):
        super().__init__(stop_criterion, step_optimizer)

    def _get_pk(self, f, xk):
        return self._grad(xk)

    def optimize(self, f: Callable, x0: np.ndarray):
        self._grad = agrad(f)
        return super().optimize(f, x0)
