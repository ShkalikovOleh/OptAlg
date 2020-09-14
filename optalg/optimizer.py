from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def optimize(self, f):
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


class OptimizerWithHistory(Optimizer):

    def __init__(self):
        super().__init__()
        self._history = []

    def get_last_history(self):
        return self._history

    def history_reset(self):
        self._history = []