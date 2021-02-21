import numpy as np
from typing import Callable, List, Generator
from abc import abstractmethod
from ...unconstrained.descent.descent_base import DescentOptimizerBase
from ...optimizer import OptimizeResult, Optimizer


class PenaltyBase(Optimizer):

    def __init__(self, unc_optimizer: DescentOptimizerBase, epsilon: float) -> None:
        self._unc_opt = unc_optimizer
        self._epsilon = epsilon

    @abstractmethod
    def _get_P(self, xk: np.ndarray, eq_constraints: List[Callable],
               ineq_constraints: List[Callable]) -> Callable:
        pass

    def optimize(self, f: Callable, x0: np.ndarray,
                 eq_constraints: List[Callable] = [],
                 ineq_constraints: List[Callable] = []) -> OptimizeResult:
        xk = x0

        iter = 0
        history = []
        history.append(xk)

        def P(x):
            return self._epsilon + 1  # for initial check

        while np.linalg.norm(P(xk)) > self._epsilon:
            P = self._get_P(xk, eq_constraints, ineq_constraints)

            def F(x):
                return f(x) + P(x)

            res = self._unc_opt.optimize(F, xk)

            xk = res.x
            history.extend(res.x_history[1:])
            iter += 1

        res = OptimizeResult(f=f, x=xk,
                             n_iter=iter,
                             n_unc_opt_iter=len(history),
                             equality_constraints=eq_constraints,
                             inequality_constraints=ineq_constraints,
                             x_history=np.array(history))

        return res


class CustomizablePenaltyBase(PenaltyBase):

    def __init__(self, unc_optimizer: DescentOptimizerBase,
                 r_eq_generator: Generator[float, None, None],
                 r_ineq_generator: Generator[float, None, None],
                 eq_penalfty_func: Callable,
                 ineq_penalty_func: Callable,
                 epsilon: float) -> None:

        super().__init__(unc_optimizer, epsilon)
        self._eq_penalty_func = eq_penalfty_func
        self._ineq_penalty_func = ineq_penalty_func
        self._r_eq_gen = r_eq_generator
        self._r_ineq_gen = r_ineq_generator

    def optimize(self, f: Callable, x0: np.ndarray,
                 eq_constraints: List[Callable] = [],
                 ineq_constraints: List[Callable] = []) -> OptimizeResult:

        self._r_eq_generator = self._r_eq_gen()
        self._r_ineq_generator = self._r_ineq_gen()
        return super().optimize(f, x0, eq_constraints, ineq_constraints)
