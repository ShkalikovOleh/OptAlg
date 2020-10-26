import numpy as np
from .newton_base import NewtonBase


class DFP(NewtonBase):

    def __init__(self, x0, stop_ctiterion, step_optimizer, renewal_step=None) -> None:
        super().__init__(x0, stop_ctiterion, step_optimizer)
        if renewal_step is None:
            renewal_step = 10000
        self.__renewal_step = renewal_step

    @property
    def renewal_step(self):
        return self.__renewal_step

    @renewal_step.setter
    def renewal_step(self, value):
        self.__renewal_step = value

    def _get_inverse_h(self, xk: np.ndarray) -> np.ndarray:
        if len(self._phistory) % self.__renewal_step == 0:
            return np.eye(xk.shape[0])

        yk = self._grad(xk) - self._pgrad
        pk = self._ahistory[-1] * self._phistory[-1]
        hk = self._phinv

        ppT = np.dot(pk, pk.T)
        yyT = np.dot(yk, yk.T)
        pTy = np.dot(pk.T, yk)

        a = ppT / pTy
        b = hk @ yyT @ hk / np.dot(np.dot(yk.T, hk), yk)

        return hk + a - b
