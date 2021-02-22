import unittest
import numpy as np
from optalg.line_search import ArmijoBacktracking
from optalg.unconstrained.descent import HestenesStiefel, FletcherReeves, PolakRibier, DaiYuan
from optalg.stop_criteria import GradientNormCriterion
from optalg.stop_criteria import IterationNumberCriterion
from ....inrange_assertion import InRangeAssertion


class ConjugateGradientsDescentTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def setUp(self):
        self.__x0 = np.array([0, 0])
        self.__gnCriterion = GradientNormCriterion(10**-3)
        self.__step_opt = ArmijoBacktracking(1, 0.5)
        self.__n = 3
        self.__opt = np.array([3, 1])

    def test_fletcher_reeves(self):
        opt = FletcherReeves(self.__gnCriterion,
                             self.__step_opt, self.__n)
        res = opt.optimize(self.f, self.__x0)

        self.assertInRange(res.x, self.__opt, 10**-3)

    def test_hestenes_stiefel(self):
        opt = HestenesStiefel(self.__gnCriterion,
                              self.__step_opt, self.__n)
        res = opt.optimize(self.f, self.__x0)

        self.assertInRange(res.x, self.__opt, 10**-3)

    def test_polak_ribier(self):
        opt = PolakRibier(self.__gnCriterion,
                          self.__step_opt, self.__n)
        res = opt.optimize(self.f, self.__x0)

        self.assertInRange(res.x, self.__opt, 10**-3)

    def test_dai_yuan(self):
        opt = DaiYuan(self.__gnCriterion,
                      self.__step_opt, self.__n)
        res = opt.optimize(self.f, self.__x0)

        self.assertInRange(res.x, self.__opt, 10**-3)
