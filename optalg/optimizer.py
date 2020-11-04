from abc import ABC, abstractmethod
from typing import Callable, Any
import numpy as np


class OptimizeResult(dict):

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except:
            raise AttributeError(f"Incorrect attribute name: {name}")


class Optimizer(ABC):

    @abstractmethod
    def optimize(self, f: Callable) -> OptimizeResult:
        """
        Optimize function

        Args:
            f (callable): function for optimize

        Example:
            def f(x):
                return x**2 + 4

            optimizer = SomeOptimizer(parameters ...)
            xmin = optimizer.optimize(f)
        """
        pass
