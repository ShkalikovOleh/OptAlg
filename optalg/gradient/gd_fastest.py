from autograd import elementwise_grad as egrad
from ..optimizer import OptimizerWithHistory
from .gradient_descent import GradientDescentOptimizer


class GradientDescentFastest(GradientDescentOptimizer):

    def __init__(self, x0, stop_criteria, step_optimizer):
        super().__init__(x0, stop_criteria)
        self.__step_optimizer = step_optimizer

    def optimize(self, f):
        grad = egrad(f)

        xk = self._x0
        self._history = [xk]

        while not self._stop_criteria.match(f, xk, self._history[-1]):
            grad_value = grad(xk)

            a = self.__step_optimizer.optimize(lambda a: f(xk - a * grad_value))

            xk = xk - a * grad_value
            self._history.append(xk)

        return xk
