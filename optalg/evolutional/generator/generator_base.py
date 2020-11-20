import numpy as np
from abc import ABC, abstractmethod


class Generator(ABC):

    @abstractmethod
    def __call__(self, population_size: int, n_variables: int) -> np.ndarray:
        pass
