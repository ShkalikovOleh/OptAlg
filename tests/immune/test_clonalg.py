import unittest
import numpy as np
from optalg.immune import ClonAlg
from ..inrange_assertion import InRangeAssertion


class ClonAlgTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return -20 * np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(
            0.5*(np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20

    def test_convergence(self):
        opt = ClonAlg(2, ((-5, 5), (-5, 5)))
        x_opt = opt.optimize(self.f)
        self.assertInRange(x_opt, np.array([0, 0]), 10**-1)

    def test_history(self):
        opt = ClonAlg(2, ((-5, 5), (-5, 5)),
                      n_generations=15, population_size=7)
        x_opt = opt.optimize(self.f)

        hist = opt.history
        self.assertEqual(np.linalg.norm(hist[-1, 0, :] - x_opt), 0)
        self.assertEqual(hist.shape, (16, 7, 2))
