import unittest
import numpy as np
from optalg.descent import Broyden
from optalg.step import BisectionWolfe, GridSearch
from optalg.stop_criteria import GradientNormCriterion, IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class BroydenTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return (x[0]**2 - x[1])**2 + (x[0] - 1)**2

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = BisectionWolfe()

        opt = Broyden(np.array([-1, -2]), gnCriterion, step_opt)
        res = opt.optimize(self.f)

        self.assertInRange(res.x, np.array([1, 1]), 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = GridSearch((10**-3, 5), 100)

        opt = Broyden(np.array([-3, -4]), nCriterion, step_opt)
        res = opt.optimize(self.f)

        hist = res.x_history
        self.assertEqual(iteration_count, hist.shape[0] - 1)
        self.assertEqual(2, hist.shape[1])