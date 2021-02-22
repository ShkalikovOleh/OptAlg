import unittest
import numpy as np
from optalg.unconstrained import NelderMead
from optalg.stop_criteria import StdFunctionNormCriterion, IterationNumberCriterion
from ..inrange_assertion import InRangeAssertion


class NelderMeadTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def test_convergence(self):
        gnCriterion = StdFunctionNormCriterion(10 ** -3)

        opt = NelderMead(gnCriterion)

        x0 = np.array([-2, -2])
        res = opt.optimize(self.f, NelderMead.generate_initial_simplex(x0, 2))

        self.assertInRange(res.x, np.array([3, 1]), 10**-3)
