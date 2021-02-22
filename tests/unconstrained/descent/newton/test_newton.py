import unittest
import numpy as np
from optalg.unconstrained.descent import Newton
from optalg.line_search import FixedStep
from optalg.stop_criteria import IterationNumberCriterion, ArgumentNormCriterion
from ....inrange_assertion import InRangeAssertion


class NewtonTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def test_convergence(self):
        gnCriterion = ArgumentNormCriterion(10**-4)
        step_opt = FixedStep(0.2)

        opt = Newton(gnCriterion, step_opt)
        res = opt.optimize(self.f, np.array([-3.5, -4]))

        self.assertInRange(res.x, np.array([3, 1]), 10**-3)
