import unittest
from optalg.iterative import Fibonacci, ModFibonacci

def f(x):
    return x**2 - 4*x + 4

# [(f, x_opt)]
funcs = [(f,2.0)]

class AdvancedIterativeTests(unittest.TestCase):
    precisions = [1,0.5,0.01,0.001,0.0001]
    bounds = [-5,5]
    

    def test_fibonacci(self):
        for prec in self.precisions:
            x = Fibonacci(self.bounds, prec).optimize(funcs[0][0])
            self.assertAlmostEqual(x, funcs[0][1], delta=prec)


    def test_modfibonacci(self):
        for prec in self.precisions:
            x = ModFibonacci(self.bounds, prec).optimize(funcs[0][0])
            self.assertAlmostEqual(x, funcs[0][1], delta=prec)
