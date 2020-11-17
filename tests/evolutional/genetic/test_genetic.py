import unittest
import numpy as np
from optalg.evolutional.genetic import Genetic
from optalg.evolutional.operators import GrayDecoder, PointCrossover, RouletteWheel, InverseMutation
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class GeneticTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        # return -20 * np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(
        #    0.5*(np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20
        return np.exp(-(x[0]**2 + x[1]**2))

    def test_convergence(self):
        n_criterion = IterationNumberCriterion(20)
        opt = Genetic(n_criterion, 2, 60,
                      GrayDecoder((-15, 15)),
                      RouletteWheel(),
                      PointCrossover(1, 0.7),
                      InverseMutation(0.07))


        for i in range(20):
            res = opt.optimize(self.f)
            self.assertInRange(res.x, np.array([[0], [0]]), 6*10**-1)
