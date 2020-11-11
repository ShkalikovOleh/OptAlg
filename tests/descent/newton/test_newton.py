import unittest
import numpy as np
from optalg.descent import Newton
from optalg.step import GridSearch, FixedStep
from optalg.stop_criteria import IterationNumberCriterion, ArgumentNormCriterion
from ...inrange_assertion import InRangeAssertion


class NewtonTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def test_convergence(self):
        gnCriterion = ArgumentNormCriterion(10**-4)
        step_opt = FixedStep(0.2)

        opt = Newton(np.array([-3.5, -4]), gnCriterion, step_opt)
        res = opt.optimize(self.f)

        self.assertInRange(res.x, np.array([3,1]), 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = GridSearch((10**-3, 5), 100)

        opt = Newton(np.array([-3, -4]), nCriterion, step_opt)
        res = opt.optimize(self.f)

        hist = res.x_history
        self.assertEqual(iteration_count, hist.shape[0] - 1)
        self.assertEqual(2, hist.shape[1])
