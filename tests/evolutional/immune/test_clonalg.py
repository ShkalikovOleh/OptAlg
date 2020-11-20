import unittest
import numpy as np
from optalg.evolutional.immune import ClonAlg
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class ClonAlgTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return -20 * np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(
            0.5*(np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20

    def test_convergence(self):
        criterion = IterationNumberCriterion(30)
        opt = ClonAlg(2, 10, criterion, [(-5, 5), (-5, 5)])
        res = opt.optimize(self.f)
        self.assertInRange(res.x, np.array([[0], [0]]), 10**-1)

    def test_history(self):
        criterion = IterationNumberCriterion(15)
        opt = ClonAlg(2, 7, criterion, [(-5, 5), (-5, 5)])
        res = opt.optimize(self.f)

        hist = res.x_history
        self.assertEqual(hist.shape, (16, 7, 2))
