import unittest
import numpy as np
from optalg.gradient import GradientDescentStepDecrease
from optalg.stop_criteria import GradientNormCriterion
from optalg.stop_criteria import IterationNumberCriterion
from ..inrange_assertion import InRangeAssertion


class GradientDescentStepDecreaseTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return 5*x**2 - 2*x + 4

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)

        opt = GradientDescentStepDecrease(np.array([-3]), 1, 0.5, gnCriterion)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, 0.2, 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)

        opt = GradientDescentStepDecrease(np.array([-3]), 1, 0.5, nCriterion)
        x_opt = opt.optimize(self.f)

        self.assertEqual(iteration_count, opt.get_last_history().shape[0] - 1)
