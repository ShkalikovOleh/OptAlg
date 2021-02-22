import numpy as np
from autograd import grad as agrad
from typing import Callable, List
from ..descent_base import DescentOptimizerBase
from ....collector import CollectorBase
from ....line_search.line_searcher import LineSearcher
from ....stop_criteria import StopCriterion


class GradientDescent(DescentOptimizerBase):

    def __init__(self, stop_criterion: StopCriterion,
                 step_optimizer: LineSearcher,
                 x_collectors: List[CollectorBase] = None,
                 direction_collectors: List[CollectorBase] = None,
                 step_collectors: List[CollectorBase] = None):

        super().__init__(stop_criterion, step_optimizer,
                         x_collectors, direction_collectors, step_collectors)

    def _get_pk(self, f, xk):
        return self._grad(xk)

    def optimize(self, f: Callable, x0: np.ndarray):
        self._grad = agrad(f)
        return super().optimize(f, x0)
