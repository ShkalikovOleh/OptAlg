import unittest
import numpy as np
from optalg.iterative import SimpleSearch
from optalg.gradient import GradientDescentFastest
from optalg.stop_criteria import GradientNormCriterion
from optalg.stop_criteria import IterationNumberCriterion
from ..inrange_assertion import InRangeAssertion


class GradientDescentFastestTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return 5*x**2 - 2*x + 4

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10 ** -3)
        step_opt = SimpleSearch((10**-3, 1), 50)

        opt = GradientDescentFastest(np.array([-3]), gnCriterion, step_opt)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, 0.2, 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = SimpleSearch((10**-3, 1), 50)

        opt = GradientDescentFastest(np.array([-3]), nCriterion, step_opt)
        x_opt = opt.optimize(self.f)

        self.assertEqual(iteration_count, opt.get_last_history().shape[0] - 1)
