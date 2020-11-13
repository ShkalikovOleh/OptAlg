from abc import ABC, abstractmethod
import numpy as np
from autograd import elementwise_grad as egrad


class NonIncompatibleStopCriterionException(Exception):
    def __str__(self) -> str:
        return "Stop criterion do not compatible with this optimizer"


class StopCriterion(ABC):

    @abstractmethod
    def match(self, f, x_history):
        pass

    def reset(self):
        pass


class IterationNumberCriterion(StopCriterion):

    def __init__(self, n: int):
        assert(n > 0)
        super().__init__()
        self.__n = n

    def match(self, f, x_history):
        if self.__n <= len(x_history) - 1:
            self.reset()
            return True
        else:
            return False


class NormCriterion(StopCriterion):

    def __init__(self, epsilon: float):
        assert(epsilon > 0)
        super().__init__()
        self._epsilon = epsilon


class GradientNormCriterion(NormCriterion):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def match(self, f, x_history):
        if len(x_history) == 0:
            return False
        elif x_history[-1].ndim == 1 or x_history[-1].shape[1] != 1 or x_history[-1].ndim > 2:
            raise NonIncompatibleStopCriterionException()

        grad = egrad(f)
        if np.linalg.norm(grad(x_history[-1])) > self._epsilon:
            return False
        else:
            return True


class ArgumentNormCriterion(NormCriterion):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def match(self, f, x_history):
        if len(x_history) < 2:
            return False
        elif isinstance(x_history[-1], np.ndarray):
            if x_history[-1].ndim > 2:
                raise NonIncompatibleStopCriterionException()
            elif x_history[-1].ndim == 2 and x_history[-1].shape[1] != 1:
                raise NonIncompatibleStopCriterionException()

        if np.linalg.norm(x_history[-1] - x_history[-2]) > self._epsilon:
            return False
        else:
            return True


class FunctionNormCriterion(NormCriterion):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def match(self, f, x_history):
        if len(x_history) < 2:
            return False
        elif isinstance(x_history[-1], np.ndarray):
            if x_history[-1].ndim > 2:
                raise NonIncompatibleStopCriterionException()
            elif x_history[-1].ndim == 2 and x_history[-1].shape[1] != 1:
                raise NonIncompatibleStopCriterionException()

        if np.linalg.norm(f(x_history[-1]) - f(x_history[-2])) > self._epsilon:
            return False
        else:
            return True


class CycleCriterion(NormCriterion):

    def __init__(self, epsilon) -> None:
        super().__init__(epsilon)

    def match(self, f, x_history):
        if len(x_history) < 3:
            return False
        elif isinstance(x_history[-1], np.ndarray):
            if x_history[-1].ndim > 2:
                raise NonIncompatibleStopCriterionException()
            elif x_history[-1].ndim == 2 and x_history[-1].shape[1] != 1:
                raise NonIncompatibleStopCriterionException()

        if np.linalg.norm(f(x_history[-1]) - f(x_history[-3])) > self._epsilon:
            return False
        else:
            return True


class CompoundCriterion(StopCriterion):

    def __init__(self, criteria_list):
        super().__init__()
        self._criteria = criteria_list


class OrCriterion(CompoundCriterion):

    def __init__(self, criteria_list):
        super().__init__(criteria_list)

    def match(self, f, x_history):
        match_result = [criterion.match(f, x_history)
                        for criterion in self._criteria]
        res = np.logical_or.reduce(match_result)

        if res:
            for criterion in self._criteria:
                criterion.reset()

        return res


class AndCriterion(CompoundCriterion):

    def __init__(self, criteria_list):
        super().__init__(criteria_list)

    def match(self, f, x_history):
        match_result = [criterion.match(f, x_history)
                        for criterion in self._criteria]
        res = np.logical_and.reduce(match_result)

        if res:
            for criterion in self._criteria:
                criterion.reset()

        return res
