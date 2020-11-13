import unittest
import numpy as np
from optalg.descent import HookeJeeves
from optalg.step import FixedStep, GridSearch
from optalg.stop_criteria import ArgumentNormCriterion, IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class HookeJeevesTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def test_convergence(self):
        argCriterion = ArgumentNormCriterion(10**-2)
        step_opt = GridSearch((0.1, 20), 100)

        opt = HookeJeeves(argCriterion, step_opt, np.array([0.05, 0.05]), 0.5)
        res = opt.optimize(self.f, np.array([-3.5, -4]))

        self.assertInRange(res.x, np.array([3,1]), 10**-2)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = FixedStep(2)

        opt = HookeJeeves(nCriterion, step_opt, np.array([0.05, 0.05]), 0.5)
        res = opt.optimize(self.f, np.array([-3.5, -4]))

        hist = res.x_history
        self.assertEqual(iteration_count, hist.shape[0] - 1)
        self.assertEqual(2, hist.shape[1])
