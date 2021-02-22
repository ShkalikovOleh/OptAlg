import unittest
import numpy as np
from optalg.unconstrained.descent import SR1
from optalg.line_search import BisectionWolfe
from optalg.stop_criteria import GradientNormCriterion, IterationNumberCriterion
from ....inrange_assertion import InRangeAssertion


class SR1Tests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return (x[0]**2 - x[1])**2 + (x[0] - 1)**2

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = BisectionWolfe()

        opt = SR1(gnCriterion, step_opt)
        res = opt.optimize(self.f, np.array([-1, -2]))

        self.assertInRange(res.x, np.array([1, 1]), 10**-3)
