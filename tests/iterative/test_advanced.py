import unittest
from optalg.iterative import Fibonacci


class AdvancedIterativeTests(unittest.TestCase):

    def test_fibonacci(self):
        precisions = [1,0.5,0.01,0.001,0.0001]
        bounds = [-5,5]

        def f(x):
            return x**2 - 4*x + 4

        for prec in precisions:
            prec = precisions[0]
            self.assertAlmostEqual( Fibonacci(bounds, prec).optimize(f), 2, delta=prec )
