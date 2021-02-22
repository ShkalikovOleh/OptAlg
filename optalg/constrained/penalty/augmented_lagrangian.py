import numpy as np
from typing import Callable, List, Generator
from ...optimizer import OptimizeResult, Optimizer
from .penalty_base import CustomizablePenaltyBase
from .utils import *


class AugmentedLagrangian(CustomizablePenaltyBase):

    def __init__(self, unc_optimizer: Optimizer,
                 r_eq_generator: Generator[float,
                                           None, None] = r_generator(1, 10),
                 r_ineq_generator: Generator[float,
                                             None, None] = r_generator(0.5, 0.1),
                 eq_penalfty_func: Callable = lambda x: x**2,
                 ineq_penalty_func: Callable = lambda x: x**2,
                 epsilon: float = 10**-3) -> None:

        if not check_sequence_increase(r_eq_generator):
            raise ValueError("R equality multipliers must increase")
        if check_sequence_increase(r_ineq_generator):
            raise ValueError("R inequality multipliers must decrease")
        if not check_function_increase(eq_penalfty_func):
            raise ValueError("Equality penalty must increase")
        if not check_function_increase(ineq_penalty_func):
            raise ValueError("Inequality penalty must increase")

        super().__init__(unc_optimizer, r_eq_generator, r_ineq_generator,
                         eq_penalfty_func, ineq_penalty_func, epsilon)

    def _get_P(self, xk: np.ndarray, eq_constraints: List[Callable],
               ineq_constraints: List[Callable]) -> Callable:

        if self.__r_eq is not None:
            for i, constr in enumerate(eq_constraints):
                self.__a[i] = self.__a[i] + self.__r_eq * constr(xk)
        if self.__r_ineq is not None:
            for i, constr in enumerate(ineq_constraints):
                t = self.__u[i] + self.__r_ineq * constr(xk)
                if t > 0:
                    self.__u[i] = t

        self.__r_eq = next(self._r_eq_generator)
        self.__r_ineq = next(self._r_ineq_generator)

        def penalty(x):
            res = 0.0

            for i, constr in enumerate(eq_constraints):
                res = res + self.__a[i] * constr(x) + \
                    self.__r_eq * self._eq_penalty_func(constr(x))

            for i, constr in enumerate(ineq_constraints):
                t = self.__u[i] + self.__r_ineq * constr(x)
                if t > 0:
                    res = res + self.__r_ineq * self._ineq_penalty_func(t)
                res = res - self.__r_ineq * self.__u[i]**2

            return res

        return penalty

    def optimize(self, f: Callable, x0: np.ndarray,
                 eq_constraints: List[Callable] = [],
                 ineq_constraints: List[Callable] = [],
                 a0=None,
                 u0=None) -> OptimizeResult:

        if a0 is None:
            self.__a = np.zeros(shape=(len(eq_constraints),))
        elif a0.shape[0] == len(eq_constraints):
            self.__a = a0
        else:
            raise ValueError(
                "a0 should have the same lenght as equality constraints")

        if a0 is None:
            self.__u = np.zeros(shape=(len(ineq_constraints),))
        elif a0.shape[0] == len(ineq_constraints):
            self.__u = u0
        else:
            raise ValueError(
                "u0 should have the same lenght as inequality constraints")

        self.__r_eq = None
        self.__r_ineq = None
        return super().optimize(f, x0, eq_constraints, ineq_constraints)
