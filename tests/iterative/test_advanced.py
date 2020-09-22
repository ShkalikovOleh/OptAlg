import unittest
from optalg.iterative import Fibonacci


class AdvancedIterativeTests(unittest.TestCase):

    def test_Fibonacci(self):
        def f1(x):
            return x**2 - 4*x + 4

        opt = Fibonacci((-5, 5), 100)
        x_opt = opt.optimize(f1)
        self.assertAlmostEqual(x_opt, 2, delta=10**-1)
