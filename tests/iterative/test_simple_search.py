import unittest
from optalg.iterative import SimpleSearch
from ..inrange_assertion import InRangeAssertion


class SimpleSearchTest(unittest.TestCase, InRangeAssertion):

    def test_convergence(self):
        def f(x):
            return x**2 - 4*x + 4

        opt = SimpleSearch((-5, 5), 100)
        x_opt = opt.optimize(f)
        self.assertInRange(x_opt, 2, 10**-1)
