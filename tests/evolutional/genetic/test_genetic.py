import unittest
import numpy as np
from optalg.evolutional.genetic import Genetic
from optalg.evolutional.operators import GrayDecoder, PointCrossover, RouletteWheel, InverseBinaryMutation, BestMerging
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class GeneticTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return -20 * np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(
            0.5*(np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20

    def test_convergence(self):
        n_criterion = IterationNumberCriterion(30)
        opt = Genetic(n_criterion, 2, 50,
                      GrayDecoder((-5, 5)),
                      RouletteWheel(),
                      PointCrossover(1, 0.7),
                      InverseBinaryMutation(0.03),
                      BestMerging())

        res = opt.optimize(self.f)
        self.assertInRange(res.x, np.array([[0], [0]]), 6*10**-1)
