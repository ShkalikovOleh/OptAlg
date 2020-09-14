from unittest import TestCase
from optalg.stop_criteria import *


def f(x):
    return x**2


class IterationNumberCriteriaTests(TestCase):

    def test_match(self):
        criteria = IterationNumberCriteria(10)
        for i in range(10):
            self.assertFalse(criteria.match(f, 0, 0))
        self.assertTrue(criteria.match(f, 0, 0))


class GradientNormCriteriaMatch(TestCase):

    def setUp(self):
        self.criteria = GradientNormCriteria(2)

    def test_match(self):
        self.assertTrue(self.criteria.match(
            f, np.array([0.9]), np.array([0.9])))

    def test_not_match(self):
        self.assertFalse(self.criteria.match(f, np.array([2]), np.array([2])))


class ArgumentNormCriteriaTests(TestCase):

    def setUp(self):
        self.criteria_fi = ArgumentNormCriteria(1)
        self.criteria_nfi = ArgumentNormCriteria(1, False)

    def test_match(self):
        self.assertTrue(self.criteria_nfi.match(f, 0, 0.5))

        self.assertFalse(self.criteria_fi.match(f, 0, 0))
        self.assertTrue(self.criteria_fi.match(f, 0, 0.5))

    def test_not_match(self):
        self.assertFalse(self.criteria_fi.match(f, 2, 4))

        self.assertFalse(self.criteria_fi.match(f, 2, 4))
        self.assertFalse(self.criteria_fi.match(f, 2, 4))


class FunctionNormCriteriaTests(TestCase):

    def setUp(self):
        self.criteria_fi = FunctionNormCriteria(1)
        self.criteria_nfi = FunctionNormCriteria(1, False)

    def test_match(self):
        self.assertTrue(self.criteria_nfi.match(f, 0, 0.5))

        self.assertFalse(self.criteria_fi.match(f, 0, 0))
        self.assertTrue(self.criteria_fi.match(f, 0, 0.5))

    def test_not_match(self):
        self.assertFalse(self.criteria_fi.match(f, 2, 4))

        self.assertFalse(self.criteria_fi.match(f, 2, 4))
        self.assertFalse(self.criteria_fi.match(f, 2, 4))
