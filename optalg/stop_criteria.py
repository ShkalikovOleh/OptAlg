from abc import ABC, abstractmethod
import numpy as np
from autograd import grad as agrad


class NonIncompatibleStopCriterionException(Exception):
    def __str__(self) -> str:
        return "Stop criterion do not compatible with this optimizer"


class StopCriterion(ABC):

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def match(self):
        pass

    @abstractmethod
    def accept(self, f, xk):
        pass

    def reset(self):
        pass


class IterationNumberCriterion(StopCriterion):

    def __init__(self, n: int):
        assert(n > 0)
        super().__init__()

        self.__n = n

    def accept(self, f, xk):
        self.__current += 1

    def match(self):
        if self.__n < self.__current:
            self.reset()
            return True
        else:
            return False

    def reset(self):
        self.__current = 0


class NormCriterion(StopCriterion):

    def __init__(self, epsilon: float):
        assert(epsilon > 0)
        super().__init__()
        self._epsilon = epsilon


class GradientNormCriterion(NormCriterion):

    def __init__(self, epsilon: float):
        super().__init__(epsilon)

    def accept(self, f, xk):
        self.__x = xk
        if self.__f is None or self.__f != f:
            self.__f = f
            self.__grad = agrad(f)

    def match(self):
        if np.linalg.norm(self.__grad(self.__x)) > self._epsilon:
            return False
        else:
            self.reset()
            return True

    def reset(self):
        self.__f = None
        self.__x = None
        self.__grad = None


class ArgumentNormCriterion(NormCriterion):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def accept(self, f, xk):
        if self.__x1 is not None:
            self.__x2 = self.__x1
        self.__x1 = xk

    def match(self):
        if self.__x2 is None or np.linalg.norm(self.__x1 - self.__x2) > self._epsilon:
            return False
        else:
            self.reset()
            return True

    def reset(self):
        self.__x1 = None
        self.__x2 = None


class FunctionNormCriterion(NormCriterion):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def accept(self, f, xk):
        if self.__x1 is not None:
            self.__x2 = self.__x1
        if self.__f is None or self.__f != f:
            self.__f = f
        self.__x1 = xk

    def match(self):
        if self.__x2 is None or np.linalg.norm(self.__f(self.__x1) - self.__f(self.__x2)) > self._epsilon:
            return False
        else:
            self.reset()
            return True

    def reset(self):
        self.__x1 = None
        self.__x2 = None
        self.__f = None


class StdFunctionNormCriterion(NormCriterion):

    def __init__(self, epsilon) -> None:
        super().__init__(epsilon)

    def accept(self, f, xk):
        self.__xs = xk
        if self.__f is None or self.__f != f:
            self.__f = f

    def match(self):
        func_values = [self.__f(x) for x in self.__xs]
        if np.std(func_values) > self._epsilon:
            return False
        else:
            self.reset()
            return True

    def reset(self):
        self.__f = None
        self.__xs = None


class CompoundCriterion(StopCriterion):

    def __init__(self, criteria_list):
        super().__init__()
        self._criteria = criteria_list

    def accept(self, f, xk):
        for criterion in self._criteria:
            criterion.accept(f, xk)


class OrCriterion(CompoundCriterion):

    def __init__(self, criteria_list):
        super().__init__(criteria_list)

    def match(self):
        match_result = [criterion.match() for criterion in self._criteria]
        res = np.logical_or.reduce(match_result)

        if res:
            for criterion in self._criteria:
                criterion.reset()

        return res


class AndCriterion(CompoundCriterion):

    def __init__(self, criteria_list):
        super().__init__(criteria_list)

    def match(self):
        match_result = [criterion.match() for criterion in self._criteria]
        return np.logical_and.reduce(match_result)
