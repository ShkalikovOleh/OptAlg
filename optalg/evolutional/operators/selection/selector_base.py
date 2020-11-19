from typing import Callable
import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class Selector(ABC):

    @abstractmethod
    def __call__(self, f: Callable, population: np.ndarray) -> np.ndarray:
        """
        Select best parents

        Return:
            :return: parent population indexes
            :rtype: ndarray with shape (n_pop_size, n_bin_lenght, n_var, n_parents)
        """
        pass
