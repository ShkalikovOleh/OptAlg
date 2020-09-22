import unittest
from optalg.iterative import Fibonacci, ModFibonacci


class FibonacciTests(unittest.TestCase):
    @staticmethod
    def f(x):
        return x**2 - 4*x + 4
    
    def setUp(self):  
        # x_opt - real optimum
        self.x_opt = 2
        self.precisions = [1,0.5,0.01,0.001,0.0001]
        self.bounds = [-5,5]

    def test_fibonacci(self):
        for prec in self.precisions:
            x = Fibonacci(self.bounds, prec).optimize(self.f)
            self.assertAlmostEqual(x, self.x_opt, delta=prec)


    def test_modfibonacci(self):
        for prec in self.precisions:
            x = ModFibonacci(self.bounds, prec).optimize(self.f)
            self.assertAlmostEqual(x, self.x_opt, delta=prec)
