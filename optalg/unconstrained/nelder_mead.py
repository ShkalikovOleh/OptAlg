import numpy as np
from typing import Callable, List
from ..collector import CollectorBase, accept_collectors, reset_collectors
from ..stop_criteria import StopCriterion
from ..optimizer import Optimizer, OptimizeResult


class NelderMead(Optimizer):
    def __init__(self, stop_criterion: StopCriterion, a=1, gamma=2, rho=0.5, delta=0.5,
                 simplex_collectors: List[CollectorBase] = None) -> None:
        assert(a > 0)
        assert(gamma > 1)
        assert(0 < rho and 0.5 >= rho)
        assert(0 < delta < 1)

        self.__stop_criterion = stop_criterion
        self.__a = a
        self.__gamma = gamma
        self.__rho = rho
        self.__delta = delta
        self.__simpex_collectors = simplex_collectors

    def __accept_all(self, f, simplex):
        self.__stop_criterion.accept(f, simplex)
        accept_collectors(self.__simpex_collectors, simplex)

    def optimize(self, f: Callable, init_simplex: List[np.ndarray]) -> OptimizeResult:

        simplex = init_simplex
        n_iter = 0

        reset_collectors(self.__simpex_collectors)
        self.__accept_all(f, simplex)

        while not self.__stop_criterion.match():
            n_iter += 1
            simplex = sorted(simplex, key=lambda x: f(x))

            xc = np.mean(simplex[:-1], axis=0)

            xr = xc + self.__a * (xc - simplex[-1])
            if f(simplex[0]) <= f(xr) and f(xr) < f(simplex[-2]):
                simplex[-1] = xr
                self.__accept_all(f, simplex)
                continue

            if f(xr) < f(simplex[0]):
                xe = xc + self.__gamma * (xr - xc)
                if f(xe) < f(simplex[0]):
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
                self.__accept_all(f, simplex)
                continue

            if f(xr) >= f(simplex[-2]):
                xcontr = xc + self.__rho * (simplex[-1] - xc)
                if f(xcontr) < f(simplex[-1]):
                    simplex[-1] = xcontr
                    self.__accept_all(f, simplex)
                    continue

            for i, x in enumerate(simplex[1:]):
                simplex[i+1] = simplex[0] + self.__delta * (x - simplex[0])

            self.__accept_all(f, simplex)

        idx = np.argmin([f(x) for x in simplex])
        res = OptimizeResult(
            f=f, x=simplex[idx],
            n_iter=n_iter)

        return res

    @staticmethod
    def generate_initial_simplex(x0: np.ndarray, l: float) -> List[np.ndarray]:
        x0 = x0.astype('float')
        simplex = [x0]
        for i in range(x0.shape[0]):
            xi = np.copy(x0)
            xi[i] += l
            simplex.append(xi)
        return simplex
