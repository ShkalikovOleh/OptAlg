from .gd_base import SimpleGradientDescentBase


class GradientDescentStepDecrease(SimpleGradientDescentBase):

    def __init__(self, x0, alpha, beta, stop_criterion):
        super().__init__(x0, stop_criterion)
        self.__a = alpha
        self.__b = beta

    def _get_a(self, f, xk, pk):
        alphaK = self.__a
        xnew = xk - alphaK * pk

        while f(xk) <= f(xnew):
            alphaK = alphaK * self.__b
            xnew = xk - alphaK * pk

        return alphaK

