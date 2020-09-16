import unittest
import numpy as np
from optalg.iterative import SimpleSearch
from optalg.gradient import ConjugateGradientsDescent
from optalg.stop_criteria import GradientNormCriteria
from optalg.stop_criteria import IterationNumberCriteria
from ..inrange_assertion import InRangeAssertion


class ConjugateGradientsDescentTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def test_convergence(self):
        gnCriteria = GradientNormCriteria(10**-3)
        step_opt = SimpleSearch((10**-3, 1), 100)

        opt = ConjugateGradientsDescent(np.array([[0], [0]]), gnCriteria, step_opt, 3)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, np.array([[3],[1]]), 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriteria = IterationNumberCriteria(iteration_count)
        step_opt = SimpleSearch((10**-3, 1), 100)

        opt = ConjugateGradientsDescent(np.array([[0], [0]]), nCriteria, step_opt, 3)
        x_opt = opt.optimize(self.f)

        self.assertEqual(iteration_count, len(opt.get_last_history()) - 1)
