import numpy as np
from jax import grad, jit, vmap
from ..descent_base import DescentOptimizerBase


class SimpleGradientDescentBase(DescentOptimizerBase):

    def __init__(self, x0, stop_criterion, **kwargs):
        super().__init__(x0, stop_criterion, **kwargs)

    def _get_pk(self, f, xk, pprev):
        return self._grad(xk)

    def optimize(self, f):
        if self.x0.size == 1:
            self._grad = jit(vmap(grad(f)))
        else:
            self._grad = jit(grad(f))
        return np.asarray(super().optimize(f))
