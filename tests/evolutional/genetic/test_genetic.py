import unittest
import numpy as np
from optalg.evolutional.genetic import Genetic
from optalg.evolutional.operators import GiniDecoder, PointCrossover, RouletteWheel, SimpleMutation
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class GeneticTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        # return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
        # return 100*(x[1] - x[0]**2)**2 + (x[0]-1)**2
        # return -20 * np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(
        #    0.5*(np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20
        return np.exp(-(x[0]**2 + x[1]**2))

    def test_convergence(self):
        n_criterion = IterationNumberCriterion(100)
        opt = Genetic(n_criterion, 2, 30,
                      GiniDecoder((0, 16), 22),
                      RouletteWheel(),
                      PointCrossover(1, 0.7),
                      SimpleMutation(0.1))

        res = opt.optimize(self.f)
        self.assertInRange(res.x, np.array([[0], [0]]), 0.6)
