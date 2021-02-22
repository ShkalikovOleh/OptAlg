import numpy as np
from typing import Callable, List
from ..stop_criteria import StopCriterion
from ..line_search import FixedStep
from ..line_search.line_searcher import LineSearcher
from ..optimizer import Optimizer, OptimizeResult
from ..collector import CollectorBase, accept_collectors, reset_collectors


class HookeJeeves(Optimizer):
    def __init__(self, stop_criterion: StopCriterion,
                 pertubation_vector: np.ndarray,
                 step_optimizer: LineSearcher = FixedStep(2),
                 gamma: float = 0.5,
                 nmax_vector_reduction: int = 10,
                 x_collectors: List[CollectorBase] = None) -> None:

        assert(gamma < 1 and gamma > 0)
        assert(nmax_vector_reduction > 0)

        self.__stop_criterion = stop_criterion
        self.__pattern_step_optimizer = step_optimizer
        self.__pertubation_vector = pertubation_vector
        self.__gamma = gamma
        self.__nmax_vector_reduction = nmax_vector_reduction
        self.__x_collectors = x_collectors

    def __exploratory_search(self, f: Callable, xk: np.ndarray, gamma: float = 1):
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
        pk = -(xk - self.__xprev)
        a = self.__pattern_step_optimizer.optimize(f, self.__xprev, pk)
        xnew = self.__xprev - a*pk
        xnew = self.__exploratory_search(f, xnew)
        self.__xprev = xk
        return xnew

    def optimize(self, f: Callable, x0: np.ndarray) -> OptimizeResult:

        xk = x0.reshape(-1, 1)
        self.__xprev = xk

        n_iter = 0
        self.__stop_criterion.accept(f, xk)
        reset_collectors(self.__x_collectors)
        accept_collectors(self.__x_collectors, xk)

        while not self.__stop_criterion.match():

            xk = self.__exploratory_search(f, xk)
            j = 0
            while f(xk) >= f(self.__xprev) and j <= self.__nmax_vector_reduction:
                xk = self.__exploratory_search(f, self.__xprev, self.__gamma**j)
                j += 1
            if j > self.__nmax_vector_reduction:
                self.__stop_criterion.reset()
                break

            while f(xk) < f(self.__xprev):
                accept_collectors(self.__x_collectors, xk)
                n_iter += 1
                xk = self.__pattern_move(f, xk)

            xk = self.__xprev
            self.__stop_criterion.accept(f, xk)

        res = OptimizeResult(f=f, x=xk.reshape(-1),
                             n_iter=n_iter)
        return res

    @staticmethod
    def generate_pertubation_vector(n_var: int, delta: float):
        assert(n_var > 0)
        return np.ones(n_var)*delta
