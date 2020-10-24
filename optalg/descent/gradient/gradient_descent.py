from autograd import elementwise_grad as egrad
from ..descent_base import DescentOptimizerBase


class GradientDescent(DescentOptimizerBase):

    def __init__(self, x0, stop_criterion, step_optimizer):
        super().__init__(x0, stop_criterion, step_optimizer)

    def _get_pk(self, f, xk, pprev):
        return self._grad(xk)

    def optimize(self, f):
        self._grad = egrad(f)
        return super().optimize(f)
