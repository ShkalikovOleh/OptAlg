import numpy as np
from autograd import elementwise_grad as egrad
from ..optimizer import OptimizerWithHistory
from .gradient_descent import GradientDescentOptimizer


class ConjugateDirectionsDescent(GradientDescentOptimizer):
    """
    Method of conjugate directions

    Descent direction is the sum of gradient in current point
    and the weighted direction from the previous iteration
    """

    def __init__(self, x0, stop_criteria, step_optimizer):
        super().__init__(x0, stop_criteria)
        self.__step_optimizer = step_optimizer

    def optimize(self, f):
        grad = egrad(f)

        xk = self._x0
        pk = 0
        self._history = [xk, xk]

        while not self._stop_criteria.match(f, xk, self._history[-2]):
            grad_value = grad(xk)

            b = np.linalg.norm(grad_value)**2 / \
                np.linalg.norm(grad(self._history[-2]))**2
            pk = grad_value + b * pk

            a = self.__step_optimizer.optimize(
                lambda a: f(xk - a * pk))

            xk = xk - a * pk
            self._history.append(xk)

        self._history = self._history[1:]

        return xk
