import unittest
from optalg.step import Fibonacci, ModFibonacci


class FibonacciTests(unittest.TestCase):
    @staticmethod
    def f(x):
        return x**2 - 4*x

    def setUp(self):
        # res - real optimum
        self.a_opt = 0.5
        self.precisions = [0.01,0.001,0.0001]
        self.bounds = [0,5]

    def test_fibonacci(self):
        for prec in self.precisions:
            a = Fibonacci(self.bounds, prec).optimize(self.f, 1, -2)
            self.assertAlmostEqual(a.x, self.a_opt, delta=prec)


    def test_modfibonacci(self):
        for prec in self.precisions:
            a = ModFibonacci(self.bounds, prec).optimize(self.f, 1, -2)
            self.assertAlmostEqual(a.x, self.a_opt, delta=prec)
