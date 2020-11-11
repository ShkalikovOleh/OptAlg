import unittest
import numpy as np
from optalg.step import ArmijoBacktracking
from optalg.descent import GradientDescent
from optalg.stop_criteria import GradientNormCriterion
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class GradientDescentTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return 5*x**2 - 2*x + 4

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10 ** -3)
        step_opt = ArmijoBacktracking(1, 0.5)

        opt = GradientDescent(np.array([-3]), gnCriterion, step_opt)
        res = opt.optimize(self.f)

        self.assertInRange(res.x, 0.2, 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = ArmijoBacktracking(1, 0.5)

        opt = GradientDescent(np.array([-3]), nCriterion, step_opt)
        res = opt.optimize(self.f)

        self.assertEqual(iteration_count, res.x_history.shape[0] - 1)
        self.assertEqual(iteration_count, res.step_history.shape[0])
        self.assertEqual(iteration_count, res.direction_history.shape[0])
