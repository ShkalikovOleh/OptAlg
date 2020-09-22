import numpy as np
from ..optimizer import Optimizer

class Fibonacci(Optimizer):
    """
    Find argmin of one-dimensional UNIMODAL function in bounds

    Drawback: Algorithm is very sensitive to float errors
    """

    def __init__(self, bounds, epsilon):
        """
        Init iterative search object

        Args:
            bounds (array-like): bounds of search. Must contain 2 values.
                                Second value must be larger than first.
            epsilon (float): calculation precision
        """
        super().__init__()
        self.a, self.b = bounds
        self.__epsilon = epsilon

    def optimize(self, f):
        diam = self.b - self.a
        F1, F2, F3 = 1, 1, 0
        j = 1
        while not F2 < diam/self.__epsilon <= F3:
            F1, F2, F3 = F2, F3, F1+F2
            j = j+1
        m = j
        
        y = self.a + diam * F1/F3
        z = self.a + self.b - y
        if f(y) <= f(z):
            a = self.a
            b = z
        else:
            a = y
            b = self.b

        k = 0
        while k<m-1:
            k=k+1
            
            if f(y) <= f(z):
                y = a + b - y
                z = y
            else:
                y = z
                z = a + b - z
            if f(y) <= f(z):
                b = z
            else:
                a = y
        X = (a+b)/2
        
        return X
