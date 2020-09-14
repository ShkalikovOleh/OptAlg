from abc import ABC, abstractmethod
import numpy as np
from autograd import elementwise_grad as egrad


class StopCriteria(ABC):

    @abstractmethod
    def match(self, f, xk, xprev):
        pass


class IterationNumberCriteria(StopCriteria):

    def __init__(self, n):
        super().__init__()
        self.__n = n
        self.reset()

    def match(self, f, xk, xprev):
        self.__current += 1
        if self.__n < self.__current:
            self.reset()
            return True
        else:
            return False

    def reset(self):
        self.__current = 0


class NormCriteria(StopCriteria):

    def __init__(self, epsilon):
        super().__init__()
        self._epsilon = epsilon


class GradientNormCriteria(NormCriteria):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def match(self, f, xk, xprev):
        grad = egrad(f)
        if np.linalg.norm(grad(xk)) > self._epsilon:
            return False
        else:
            return True


class ArgumentNormCriteria(NormCriteria):

    def __init__(self, epsilon, detect_first_iteration=True):
        super().__init__(epsilon)
        self.detect_first_iteration = detect_first_iteration
        self.first_iteration = detect_first_iteration

    def match(self, f, xk, xprev):
        if np.linalg.norm(xk - xprev) > self._epsilon:
            return False
        elif self.first_iteration:
            self.first_iteration = False
            return False
        else:
            self.first_iteration = self.detect_first_iteration
            return True


class FunctionNormCriteria(NormCriteria):

    def __init__(self, epsilon, detect_first_iteration=True):
        super().__init__(epsilon)
        self.detect_first_iteration = detect_first_iteration
        self.first_iteration = detect_first_iteration

    def match(self, f, xk, xprev):
        if np.linalg.norm(f(xk) - f(xprev)) > self._epsilon:
            return False
        elif self.first_iteration:
            self.first_iteration = False
            return False
        else:
            self.first_iteration = self.detect_first_iteration
            return True
