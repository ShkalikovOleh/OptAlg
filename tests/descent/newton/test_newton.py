import unittest
import numpy as np
from optalg.descent import Newton
from optalg.iterative import SimpleSearch
from optalg.stop_criteria import GradientNormCriterion
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class NewtonTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = SimpleSearch((10**-3, 5), 100)

        opt = Newton(np.array([[-3.5], [-4]]), gnCriterion, step_opt)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, np.array([[3],[1]]), 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = SimpleSearch((10**-3, 5), 100)

        opt = Newton(np.array([[-3], [-4]]), nCriterion, step_opt)
        x_opt = opt.optimize(self.f)

        self.assertEqual(iteration_count, opt.get_last_history().shape[1] - 1)
