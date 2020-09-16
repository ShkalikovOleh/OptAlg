import numpy as np
from autograd import elementwise_grad as egrad
from .gradient_descent import GradientDescentOptimizer


class ConjugateGradientsDescent(GradientDescentOptimizer):
    """
    Method of conjugate gradients

    Descent direction is the sum of gradient in current point
    and the weighted direction from the previous iteration
    """

    def __init__(self, x0, stop_criteria, step_optimizer, reset_iteration_number):
        super().__init__(x0, stop_criteria)
        self.__step_optimizer = step_optimizer
        self.__reset_iteration_number = reset_iteration_number

    def optimize(self, f):
        grad = egrad(f)

        xk = self._x0
        pk = 0
        self._history = [xk, xk]

        iteration = 0
        while not self._stop_criteria.match(f, xk, self._history[-2]):
            grad_value = grad(xk)
            iteration += 1

            b = np.linalg.norm(grad_value)**2 / \
                np.linalg.norm(grad(self._history[-2]))**2

            if (self.__reset_iteration_number == iteration):  # reset pk for method convergence
                pk = 0

            pk = grad_value + b * pk

            a = self.__step_optimizer.optimize(
                lambda a: f(xk - a * pk))

            xk = xk - a * pk
            self._history.append(xk)

        self._history = self._history[1:]

        return xk
