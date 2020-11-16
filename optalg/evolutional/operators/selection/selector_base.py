from typing import Callable
import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class Selector(ABC):

    @abstractmethod
    def __call__(self, f: Callable, population: np.ndarray) -> np.ndarray:
        pass
