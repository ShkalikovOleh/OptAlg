import numpy as np
from .newton_base import NewtonBase


class SR1(NewtonBase):

    def __init__(self, x0, stop_ctiterion, step_optimizer) -> None:
        super().__init__(x0, stop_ctiterion, step_optimizer)

    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        if len(self._phistory) == 0:
            return np.eye(xk.shape[0])

        gk = self._grad(xk) - self._pgrad
        pk = xk - self._history[-2]
        hk = self._inv_hessian_history[-1]

        t = pk - hk @ gk
        numerator = np.outer(t, t)
        denominator = gk.T @ t

        return hk + numerator / denominator
