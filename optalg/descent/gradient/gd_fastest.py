from autograd import elementwise_grad as egrad
from .gd_base import SimpleGradientDescentBase


class GradientDescentFastest(SimpleGradientDescentBase):

    def __init__(self, x0, stop_criterion, step_optimizer):
        super().__init__(x0, stop_criterion)
        self.__step_opt = step_optimizer

    def _get_a(self, f, xk, pk):
        return self.__step_opt.optimize(lambda a: f(xk - a * pk))
