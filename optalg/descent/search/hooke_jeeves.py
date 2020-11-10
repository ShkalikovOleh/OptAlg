import numpy as np
from typing import Callable
from ...stop_criteria import StopCriterion
from ...step.step_optimizer import StepOptimizer
from ..descent_base import DescentOptimizerBase


class HookeJeeves(DescentOptimizerBase):
    def __init__(self, x0: np.ndarray, stop_criterion: StopCriterion,
                 step_optimizer: StepOptimizer, pertubation_vector, gamma) -> None:

        assert(gamma < 1 and gamma > 0)

        super().__init__(x0, stop_criterion, step_optimizer)
        self.__pertubation_vector = pertubation_vector
        self.__gamma = gamma

    def _get_pk(self, f: Callable, xk: np.ndarray) -> np.ndarray:
        deltas = np.zeros_like(xk)
        j = 0
        while np.linalg.norm(deltas) == 0:
            for i in range(xk.shape[0]):
                t = self.__gamma**j * self.__pertubation_vector[i]
                temp = np.zeros_like(xk)
                temp[i] = t

                xnew1 = xk + temp
                xnew2 = xk - temp

                if f(xnew1) < f(xk) and f(xnew1) <= f(xnew2):
                    deltas[i] = -t
                elif f(xnew2) < f(xk):
                    deltas[i] = t
            j += 1
        return deltas
