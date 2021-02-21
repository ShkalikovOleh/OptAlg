import numpy as np
from .newton_base import NewtonBase


class DFP(NewtonBase):

    def __init__(self, stop_ctiterion, step_optimizer) -> None:
        super().__init__(stop_ctiterion, step_optimizer)

    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        if len(self._phistory) == 0:
            return np.eye(xk.shape[0])

        gk = self._grad(xk) - self._pgrad
        pk = xk - self._history[-2]
        hk = self._inv_hessian_history[-1]

        ppT = np.outer(pk, pk)
        yyT = np.outer(gk, gk)
        pTy = np.outer(pk, gk)

        a = ppT / pTy
        b = hk @ yyT @ hk / (gk.T @ hk @ gk)

        return hk + a - b
