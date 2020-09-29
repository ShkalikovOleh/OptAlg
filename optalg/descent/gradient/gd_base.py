from autograd import elementwise_grad as egrad
from ..descent_base import DescentOptimizerBase


class SimpleGradientDescentBase(DescentOptimizerBase):

    def __init__(self, x0, stop_criterion):
        super().__init__(x0, stop_criterion)

    def _get_pk(self, f, xk, pprev):
        return self._grad(xk)

    def optimize(self, f):
        self._grad = egrad(f)
        return super().optimize(f)
