import unittest
import numpy as np
from optalg.unconstrained.descent import DFP
from optalg.line_search import ArmijoBacktracking
from optalg.stop_criteria import GradientNormCriterion, IterationNumberCriterion
from ....inrange_assertion import InRangeAssertion


class DFPTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return (x[0]**2 - x[1])**2 + (x[0] - 1)**2

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = ArmijoBacktracking(1, 0.5)

        opt = DFP(gnCriterion, step_opt)
        res = opt.optimize(self.f, np.array([-1, -2]))

        self.assertInRange(res.x, np.array([1, 1]), 10**-2)
