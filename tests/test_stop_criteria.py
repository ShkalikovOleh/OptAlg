from unittest import TestCase
from optalg.stop_criteria import *


def f(x):
    return x**2


class IterationNumberCriterionTests(TestCase):

    def test_match(self):
        Criterion = IterationNumberCriterion(10)
        for i in range(10):
            self.assertFalse(Criterion.match())
            Criterion.accept(f, np.array([1]))

        Criterion.accept(f, np.array([1]))
        self.assertTrue(Criterion.match())


class GradientNormCriterionMatch(TestCase):

    def setUp(self):
        self.Criterion = GradientNormCriterion(2)

    def test_match(self):
        self.Criterion.accept(f, np.array([0.9]).reshape(-1, 1))
        self.assertTrue(self.Criterion.match())

    def test_not_match(self):
        self.Criterion.accept(f, np.array([2]).reshape(-1, 1))
        self.assertFalse(self.Criterion.match())


class ArgumentNormCriterionTests(TestCase):

    def setUp(self):
        self.Criterion = ArgumentNormCriterion(1)

    def test_match(self):
        self.Criterion.accept(f, np.array([0, 0.5]))
        self.Criterion.accept(f, np.array([0.5, 0]))
        self.assertTrue(self.Criterion.match())

    def test_not_match(self):
        self.Criterion.accept(f, np.array([0, 2]))
        self.assertFalse(self.Criterion.match())
        self.Criterion.accept(f, np.array([0, -2]))
        self.assertFalse(self.Criterion.match())


class FunctionNormCriterionTests(TestCase):

    def setUp(self):
        self.Criterion=FunctionNormCriterion(1)

    def test_match(self):
        self.Criterion.accept(f, np.array([0.9]))
        self.Criterion.accept(f, np.array([0.9]))
        self.assertTrue(self.Criterion.match())

    def test_not_match(self):
        self.Criterion.accept(f, np.array([2]))
        self.assertFalse(self.Criterion.match())
        self.Criterion.accept(f, np.array([4]))
        self.assertFalse(self.Criterion.match())


class AndCriterionTests(TestCase):

    def test_match(self):
        criterion=AndCriterion([IterationNumberCriterion(1),
                                  ArgumentNormCriterion(10**-3)])
        criterion.accept(f, np.array([0.9]))
        criterion.accept(f, np.array([0.9]))
        self.assertTrue(criterion.match())

    def test_not_match(self):
        criterion=AndCriterion([IterationNumberCriterion(1),
                                  ArgumentNormCriterion(10**-3)])
        criterion.accept(f, np.array([2]))
        criterion.accept(f, np.array([2.5]))
        self.assertFalse(criterion.match())


class OrCriterionTests(TestCase):

    def test_match(self):
        criterion=OrCriterion([IterationNumberCriterion(3),
                                 ArgumentNormCriterion(10**-3)])
        criterion.accept(f, np.array([2]))
        criterion.accept(f, np.array([2]))
        self.assertTrue(criterion.match())

    def test_not_match(self):
        criterion=OrCriterion([IterationNumberCriterion(3),
                                 ArgumentNormCriterion(10**-3)])
        criterion.accept(f, np.array([2]))
        criterion.accept(f, np.array([3]))
        self.assertFalse(criterion.match())
