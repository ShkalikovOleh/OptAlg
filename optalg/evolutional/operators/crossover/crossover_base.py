import numpy as np
from abc import ABC, abstractmethod

class CrossoverOperator(ABC):

    @abstractmethod
    def __call__(self, mating_genotypes: np.ndarray) -> np.ndarray:
        pass
