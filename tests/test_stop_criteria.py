from unittest import TestCase
from optalg.stop_criteria import *


def f(x):
    return x**2


class IterationNumberCriterionTests(TestCase):

    def test_match(self):
        Criterion = IterationNumberCriterion(10)
        for i in range(10):
            self.assertFalse(Criterion.match(f, 0, 0))
        self.assertTrue(Criterion.match(f, 0, 0))


class GradientNormCriterionMatch(TestCase):

    def setUp(self):
        self.Criterion = GradientNormCriterion(2)

    def test_match(self):
        self.assertTrue(self.Criterion.match(
            f, np.array([0.9]), np.array([0.9])))

    def test_not_match(self):
        self.assertFalse(self.Criterion.match(f, np.array([2]), np.array([2])))


class ArgumentNormCriterionTests(TestCase):

    def setUp(self):
        self.Criterion_fi = ArgumentNormCriterion(1)
        self.Criterion_nfi = ArgumentNormCriterion(1, False)

    def test_match(self):
        self.assertTrue(self.Criterion_nfi.match(f, 0, 0.5))

        self.assertFalse(self.Criterion_fi.match(f, 0, 0))
        self.assertTrue(self.Criterion_fi.match(f, 0, 0.5))

    def test_not_match(self):
        self.assertFalse(self.Criterion_fi.match(f, 2, 4))

        self.assertFalse(self.Criterion_fi.match(f, 2, 4))
        self.assertFalse(self.Criterion_fi.match(f, 2, 4))


class FunctionNormCriterionTests(TestCase):

    def setUp(self):
        self.Criterion_fi = FunctionNormCriterion(1)
        self.Criterion_nfi = FunctionNormCriterion(1, False)

    def test_match(self):
        self.assertTrue(self.Criterion_nfi.match(f, 0, 0.5))

        self.assertFalse(self.Criterion_fi.match(f, 0, 0))
        self.assertTrue(self.Criterion_fi.match(f, 0, 0.5))

    def test_not_match(self):
        self.assertFalse(self.Criterion_fi.match(f, 2, 4))

        self.assertFalse(self.Criterion_fi.match(f, 2, 4))
        self.assertFalse(self.Criterion_fi.match(f, 2, 4))


class AndCriterionTests(TestCase):

    def test_match(self):
        criterion = AndCriterion([IterationNumberCriterion(
            0), ArgumentNormCriterion(10**-3, False)])
        self.assertTrue(criterion.match(
            f, np.array([0.9]), np.array([0.9])))

    def test_not_match(self):
        criterion = AndCriterion([IterationNumberCriterion(0),
                                ArgumentNormCriterion(10**-3, True)])
        self.assertFalse(criterion.match(f, np.array([2]), np.array([2])))


class OrCriterionTests(TestCase):

    def test_match(self):
        criterion = OrCriterion([IterationNumberCriterion(0),
                                ArgumentNormCriterion(10**-3, True)])
        self.assertTrue(criterion.match(f, np.array([2]), np.array([2])))

    def test_not_match(self):
        criterion = OrCriterion([IterationNumberCriterion(1),
                                ArgumentNormCriterion(10**-3, True)])
        self.assertFalse(criterion.match(f, np.array([2]), np.array([2])))
