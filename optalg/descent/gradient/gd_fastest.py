from autograd import elementwise_grad as egrad
from ..descent_base import FastestDescentBase
from .gd_base import SimpleGradientDescentBase


class GradientDescentFastest(SimpleGradientDescentBase, FastestDescentBase):

    def __init__(self, x0, stop_criterion, step_optimizer):
        super().__init__(x0=x0, stop_criterion=stop_criterion, step_opt=step_optimizer)
