from abc import ABC, abstractmethod
import numpy as np
from jax import grad, vmap


class StopCriterion(ABC):

    @abstractmethod
    def match(self, f, xk, xprev):
        pass

    def reset(self):
        pass


class IterationNumberCriterion(StopCriterion):

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


class NormCriterion(StopCriterion):

    def __init__(self, epsilon):
        super().__init__()
        self._epsilon = epsilon


class GradientNormCriterion(NormCriterion):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def match(self, f, xk, xprev):
        if xk.size == 1:
            grad_fn = vmap(grad(f))
        else:
            grad_fn = grad(f)

        if np.linalg.norm(grad_fn(xk)) > self._epsilon:
            return False
        else:
            return True


class ArgumentNormCriterion(NormCriterion):

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
            self.reset()
            return True

    def reset(self):
        self.first_iteration = self.detect_first_iteration


class FunctionNormCriterion(NormCriterion):

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
            self.reset()
            return True

    def reset(self):
        self.first_iteration = self.detect_first_iteration


class CompoundCriterion(StopCriterion):

    def __init__(self, criteria_list):
        super().__init__()
        self._criteria = criteria_list


class OrCriterion(CompoundCriterion):

    def __init__(self, criteria_list):
        super().__init__(criteria_list)

    def match(self, f, xk, xprev):
        match_result = [criterion.match(f, xk, xprev)
                        for criterion in self._criteria]
        res = np.logical_or.reduce(match_result)

        if res:
            for criterion in self._criteria:
                criterion.reset()

        return res


class AndCriterion(CompoundCriterion):

    def __init__(self, criteria_list):
        super().__init__(criteria_list)

    def match(self, f, xk, xprev):
        match_result = [criterion.match(f, xk, xprev)
                        for criterion in self._criteria]
        res = np.logical_and.reduce(match_result)

        if res:
            for criterion in self._criteria:
                criterion.reset()

        return res
