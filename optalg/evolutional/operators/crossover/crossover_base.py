import numpy as np
from abc import ABC, abstractmethod

class CrossoverOperator(ABC):

    @abstractmethod
    def __call__(self, population: np.ndarray, parents_idx: np.ndarray) -> np.ndarray:
        pass
