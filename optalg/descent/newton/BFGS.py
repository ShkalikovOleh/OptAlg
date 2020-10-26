import numpy as np
from .newton_base import NewtonBase


class BFGS(NewtonBase):

    def __init__(self, x0, stop_ctiterion, step_optimizer) -> None:
        super().__init__(x0, stop_ctiterion, step_optimizer)

    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        I = np.eye(xk.shape[0])

        if len(self._phistory) == 0:
            return I

        yk = self._grad(xk) - self._pgrad
        pk = self._ahistory[-1] * self._phistory[-1]
        hk = self._phinv

        yTp = np.dot(yk.T, pk)
        ypT = np.dot(yk, pk.T)
        pyT = np.dot(pk, yk.T)

        a = (I - pyT/yTp) @ hk @ (I - ypT/yTp)
        b = pyT / yTp

        return a + b
