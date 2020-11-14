from unittest import TestCase
from optalg.stop_criteria import *


def f(x):
    return x**2


class IterationNumberCriterionTests(TestCase):

    def test_match(self):
        Criterion = IterationNumberCriterion(10)
        t_exp = [i for i in range(11)]
        f_exp = [i for i in range(10)]
        self.assertFalse(Criterion.match(f, f_exp))
        self.assertTrue(Criterion.match(f, t_exp))


class GradientNormCriterionMatch(TestCase):

    def setUp(self):
        self.Criterion = GradientNormCriterion(2)

    def test_match(self):
        self.assertTrue(self.Criterion.match(f, [np.array([0.9]).reshape(-1,1)]))

    def test_not_match(self):
        self.assertFalse(self.Criterion.match(f, [np.array([2]).reshape(-1,1)]))


class ArgumentNormCriterionTests(TestCase):

    def setUp(self):
        self.Criterion = ArgumentNormCriterion(1)

    def test_match(self):
        self.assertTrue(self.Criterion.match(f, [0, 0.5]))
        self.assertTrue(self.Criterion.match(f, [0, 0]))
        self.assertTrue(self.Criterion.match(
            f, [np.array([-1,1]).reshape(-1, 1), np.array([-1,1]).reshape(-1, 1)]))

    def test_not_match(self):
        self.assertFalse(self.Criterion.match(f, [0, 2]))
        self.assertFalse(self.Criterion.match(f, [0, -2]))
        self.assertFalse(self.Criterion.match(
            f, [np.array([-1, 1]).reshape(-1, 1), np.array([-1, 2.1]).reshape(-1, 1)]))


class FunctionNormCriterionTests(TestCase):

    def setUp(self):
        self.Criterion = FunctionNormCriterion(1)

    def test_match(self):
        self.assertTrue(self.Criterion.match(f, [0, 0.5]))
        self.assertTrue(self.Criterion.match(f, [0, 0]))

    def test_not_match(self):
        self.assertFalse(self.Criterion.match(f, [2, 4]))
        self.assertFalse(self.Criterion.match(f, [2, 4]))


class AndCriterionTests(TestCase):

    def test_match(self):
        criterion = AndCriterion([IterationNumberCriterion(
            1), ArgumentNormCriterion(10**-3)])
        self.assertTrue(criterion.match(f, [np.array([0.9]), np.array([0.9])]))

    def test_not_match(self):
        criterion = AndCriterion([IterationNumberCriterion(2),
                                ArgumentNormCriterion(10**-3)])
        self.assertFalse(criterion.match(f, [np.array([2]), np.array([2])]))


class OrCriterionTests(TestCase):

    def test_match(self):
        criterion = OrCriterion([IterationNumberCriterion(1),
                                ArgumentNormCriterion(10**-3)])
        self.assertTrue(criterion.match(f, [np.array([2]), np.array([2])]))

    def test_not_match(self):
        criterion = OrCriterion([IterationNumberCriterion(3),
                                ArgumentNormCriterion(10**-3)])
        self.assertFalse(criterion.match(f, [np.array([2]), np.array([3])]))
