import numpy as np
from autograd.numpy import log
from typing import Callable, Generator, List
from ...unconstrained.descent.descent_base import DescentOptimizerBase
from ...optimizer import OptimizeResult
from .penalty_base import CustomizablePenaltyBase
from .utils import *


class Interior(CustomizablePenaltyBase):
    """
    Barrier method for constrained optimization
    By default uses log(-x) barrier function, but
    you can specify inequality and equality
    constraints functions and r generators.
    Method must start from internal point.

    F(x) = f(x) + r_eq * eq_penalty(eq_constr(x)) - r_ineq * ineq_penalty(ineq_constr(x))
    """

    def __init__(self, unc_optimizer: DescentOptimizerBase,
                 r_eq_generator: Generator[float,
                                           None, None] = r_generator(1/2, 4),
                 r_ineq_generator: Generator[float,
                                             None, None] = r_generator(1, 1/4),
                 eq_penalfty_func: Callable = lambda x: x**2,
                 ineq_penalty_func: Callable = lambda x: log(-x),
                 epsilon: float = 10**-3) -> None:

        if not check_sequence_increase(r_eq_generator):
            raise ValueError("R equality multipliers must increase")
        if check_sequence_increase(r_ineq_generator):
            raise ValueError("R inequality multipliers must decrease")
        if not check_function_increase(eq_penalfty_func):
            raise ValueError("Equality penalty must increase")
        if check_function_increase(ineq_penalty_func):
            raise ValueError("Inequality penalty must decrease")

        super().__init__(unc_optimizer, r_eq_generator, r_ineq_generator,
                         eq_penalfty_func, ineq_penalty_func, epsilon)

    def _get_P(self, xk: np.ndarray, eq_constraints: List[Callable],
               ineq_constraints: List[Callable]) -> Callable:
        r_eq = next(self._r_eq_generator)
        r_ineq = next(self._r_ineq_generator)

        def penalty(x):
            res = 0.0

            for constr in eq_constraints:
                res = res + r_eq * self._eq_penalty_func(constr(x))

            for constr in ineq_constraints:
                res = res - r_ineq * self._ineq_penalty_func(constr(x))

            return res

        return penalty

    def optimize(self, f: Callable, x0: np.ndarray,
                 eq_constraints: List[Callable] = [],
                 ineq_constraints: List[Callable] = []) -> OptimizeResult:

        for constr in ineq_constraints:
            if constr(x0) > 0:
                raise ValueError("Starting point should be internal")

        return super().optimize(f, x0, eq_constraints, ineq_constraints)
