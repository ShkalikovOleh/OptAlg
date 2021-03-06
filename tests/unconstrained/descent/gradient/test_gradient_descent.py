import unittest
import numpy as np
from optalg.line_search import ArmijoBacktracking
from optalg.unconstrained.descent import GradientDescent
from optalg.stop_criteria import GradientNormCriterion
from optalg.stop_criteria import IterationNumberCriterion
from ....inrange_assertion import InRangeAssertion


class GradientDescentTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return 5*x**2 - 2*x + 4

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10 ** -3)
        step_opt = ArmijoBacktracking(1, 0.5)

        opt = GradientDescent(gnCriterion, step_opt)
        res = opt.optimize(self.f, np.array([-3]))

        self.assertInRange(res.x, 0.2, 10**-3)
