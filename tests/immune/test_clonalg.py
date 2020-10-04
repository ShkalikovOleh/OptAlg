import unittest
import numpy as np
from optalg.immune import ClonAlg
from ..inrange_assertion import InRangeAssertion


class ClonAlgTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 4*x[1]**2 + 4

    def test_convergence(self):

        opt = ClonAlg(2, ((-5, 5), (-5, 5)))
        x_opt = opt.optimize(self.f)
        self.assertInRange(x_opt, np.array([[0],[0]]), 10 ** -1)

    def test_history(self):
        opt = ClonAlg(2, ((-5, 5),(-5, 5)), n_generations=15, population_size=7)
        x_opt = opt.optimize(self.f)

        hist = opt.get_last_history()
        self.assertEqual(np.linalg.norm(hist[-1, 0, :] - x_opt.reshape(2)), 0)
        self.assertEqual(hist.shape, (16,7,2))
