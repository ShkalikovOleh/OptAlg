import unittest
import numpy as np
from optalg.step import GridSearch
from optalg.descent import HestenesStiefel, FletcherReeves, PolakRibier, DaiYuan
from optalg.stop_criteria import GradientNormCriterion
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class ConjugateGradientsDescentTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + 2*x[1]**2 + x[0]*x[1] - 7*x[0] - 7*x[1]

    def setUp(self):
        self.__x0 = np.array([[0], [0]])
        self.__gnCriterion = GradientNormCriterion(10**-3)
        self.__step_opt = GridSearch((10 ** -3, 1), 100)
        self.__n = 3
        self.__opt = np.array([[3], [1]])

    def test_fletcher_reeves(self):
        opt = FletcherReeves(self.__x0, self.__gnCriterion,
                              self.__step_opt, self.__n)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, self.__opt, 10**-3)

    def test_hestenes_stiefel(self):
        opt = HestenesStiefel(self.__x0, self.__gnCriterion,
                             self.__step_opt, self.__n)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, self.__opt, 10**-3)

    def test_polak_ribier(self):
        opt = PolakRibier(self.__x0, self.__gnCriterion,
                              self.__step_opt, self.__n)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, self.__opt, 10**-3)

    def test_dai_yuan(self):
        opt = DaiYuan(self.__x0, self.__gnCriterion,
                          self.__step_opt, self.__n)
        x_opt = opt.optimize(self.f)

        self.assertInRange(x_opt, self.__opt, 10**-3)

    def test_get_history(self):
        iteration_count = 10
        nCriterion = IterationNumberCriterion(iteration_count)
        step_opt = GridSearch((10**-3, 1), 100)

        opt = HestenesStiefel(np.array([[0], [0]]), nCriterion, step_opt, 3)
        x_opt = opt.optimize(self.f)

        self.assertEqual(iteration_count, opt.history.shape[0] - 1)
