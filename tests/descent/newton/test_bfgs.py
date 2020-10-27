import unittest
import numpy as np
from optalg.descent import BFGS
from optalg.step import ArmijoBacktracking, GridSearch
from optalg.stop_criteria import GradientNormCriterion, IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class BFGSTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return (x[0]**2 - x[1])**2 + (x[0] - 1)**2

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = ArmijoBacktracking(1, 0.5)

        opt = BFGS(np.array([-1, -2]), gnCriterion, step_opt)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, np.array([1, 1]), 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = GridSearch((10**-3, 5), 100)

        opt = BFGS(np.array([-3, -4]), nCriterion, step_opt)
        x_opt = opt.optimize(self.f)

        hist = opt.history
        self.assertEqual(iteration_count, hist.shape[0] - 1)
        self.assertEqual(1, hist.shape[1])
        self.assertEqual(2, hist.shape[2])
