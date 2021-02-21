import numpy as np
from typing import Callable
from .line_searcher import LineSearcher


class FixedStep(LineSearcher):

    def __init__(self, step) -> None:
        self.__step = step

    def optimize(self, f: Callable, xk: np.ndarray, pk: np.ndarray):
        return self.__step
