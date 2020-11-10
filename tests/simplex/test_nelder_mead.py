import unittest
import numpy as np
from optalg.simplex import NelderMead, generate_initial_simplex
from optalg.stop_criteria import GradientNormCriterion, IterationNumberCriterion
from ..inrange_assertion import InRangeAssertion


class NelderMeadTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10 ** -3)

        opt = NelderMead(gnCriterion)

        x0 = np.array([-2, -2])
        res = opt.optimize(self.f, generate_initial_simplex(x0, 2))

        self.assertInRange(res.x, np.array([3, 1]), 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)

        opt = NelderMead(nCriterion)

        x0 = np.array([-2, -2])
        res = opt.optimize(self.f, generate_initial_simplex(x0, 2))

        self.assertEqual(iteration_count, res.x_history.shape[0] - 1)
