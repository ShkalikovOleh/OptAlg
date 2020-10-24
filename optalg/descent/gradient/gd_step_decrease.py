from ..descent_base import StepDecreaseDescentBase
from .gd_base import SimpleGradientDescentBase


class GradientDescentStepDecrease(SimpleGradientDescentBase, StepDecreaseDescentBase):

    def __init__(self, x0, stop_criterion, alpha=1, beta=0.5):
        super().__init__(x0=x0,
                         stop_criterion=stop_criterion,
                         a=alpha, b=beta)
