import numpy as np
from typing import Callable, List
from ..stop_criteria import StopCriterion
from ..optimizer import Optimizer, OptimizeResult


class NelderMead(Optimizer):
    def __init__(self, stop_criterion: StopCriterion, a=1, gamma=2, rho=0.5, delta=0.5) -> None:
        assert(a > 0)
        assert(gamma > 1)
        assert(0 < rho and 0.5 >= rho)
        assert(0 < delta < 1)

        self.__stop_criterion = stop_criterion
        self.__a = a
        self.__gamma = gamma
        self.__rho = rho
        self.__delta = delta
        self.__history = []

    def __check_stop_criterion(self, f, simplex):
        x = min(simplex, key=lambda x: f(x))
        if len(self.__history) > 0:
            xprev = self.__history[-1][0]
        else:
            xprev = x
        return self.__stop_criterion.match(f, x, xprev)

    def optimize(self, f: Callable, init_simplex: List[np.ndarray]) -> OptimizeResult:
        if(len(init_simplex) < 3):
            raise ValueError("Initial simplex must contains >= 3 points")

        simplex = init_simplex

        while not self.__check_stop_criterion(f, simplex):
            simplex = sorted(simplex, key=lambda x: f(x))
            self.__history.append(simplex.copy())

            xc = np.mean(simplex[:-1], axis=0)

            xr = xc + self.__a * (xc - simplex[-1])
            if f(simplex[0]) <= f(xr) and f(xr) < f(simplex[-2]):
                simplex[-1] = xr
                continue

            if f(xr) < f(simplex[0]):
                xe = xc + self.__gamma * (xr - xc)
                if f(xe) < f(simplex[0]):
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
                continue

            if f(xr) >= f(simplex[-2]):
                xcontr = xc + self.__rho * (simplex[-1] - xc)
                if f(xcontr) < f(simplex[-1]):
                    simplex[-1] = xcontr
                    continue

            for i, x in enumerate(simplex[1:]):
                simplex[i+1] = simplex[0] + self.__delta * (x - simplex[0])

        simplex = sorted(simplex, key=lambda x: f(x))
        self.__history.append(simplex.copy())

        res = OptimizeResult(
            f=f, x=simplex[0], x_history=np.array(self.__history))

        self.__history.clear()
        return res


def generate_initial_simplex(x0: np.ndarray, l: float) -> List[np.ndarray]:
    x0 = x0.astype('float')
    simplex = [x0]
    for i in range(x0.shape[0]):
        xi = np.copy(x0)
        xi[i] += l
        simplex.append(xi)
    return simplex
