import numpy as np
from typing import Callable
from ...stop_criteria import StopCriterion
from ...step.step_optimizer import StepOptimizer
from ...step.fixed import FixedStep
from ..descent_base import DescentOptimizerBase


class HookeJeeves(DescentOptimizerBase):
    def __init__(self, stop_criterion: StopCriterion,
                 step_optimizer: StepOptimizer,
                 pertubation_vector: np.ndarray,
                 gamma: float = 0.5,
                 max_vector_reduction: int = 10) -> None:

        assert(gamma < 1 and gamma > 0)
        assert(max_vector_reduction > 0)

        super().__init__(stop_criterion, FixedStep(1))
        self.__pattern_step_optimizer = step_optimizer
        self.__pertubation_vector = pertubation_vector
        self.__gamma = gamma
        self.__max_vector_reduction = max_vector_reduction

    def __exploratory_search(self, f: Callable, xk: np.ndarray, gamma: int = 1):
        xnew = np.copy(xk)
        pertubations = np.diag(self.__pertubation_vector) * gamma

        for i in range(xk.shape[0]):
            delta = pertubations[:, [i]]
            xnew1 = xnew + delta
            xnew2 = xnew - delta
            if f(xnew1) < f(xnew) and f(xnew1) <= f(xnew2):
                xnew = xnew1
            elif f(xnew2) < f(xnew):
                xnew = xnew2

        return xnew

    def __pattern_move(self, f: Callable, xk: np.ndarray):
        xprev = self._history[-2]
        pk = -(xk - xprev)
        a = self.__pattern_step_optimizer.optimize(f, xprev, pk).x
        xnew = xprev - a*pk
        xnew = self.__exploratory_search(f, xnew)
        return xnew

    def _get_pk(self, f: Callable, xk: np.ndarray) -> np.ndarray:
        xnew = np.copy(xk)
        if len(self._history) >= 2:
            xnew = self.__pattern_move(f, xk)

        if f(xnew) >= f(xk):
            j = 0
            while (f(xnew) >= f(xk) or np.linalg.norm(xnew - xk) == 0) and j < self.__max_vector_reduction:
                xnew = self.__exploratory_search(f, xk, self.__gamma**j)
                j += 1

        return -(xnew - xk)

    @staticmethod
    def generate_pertubation_vector(n_var: int, delta: float):
        assert(n_var > 0)
        return np.ones(n_var)*delta
