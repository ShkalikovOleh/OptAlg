import numpy as np
from abc import ABC, abstractmethod


class MutationOperator(ABC):

    @abstractmethod
    def __call__(self, pop_genotypes) -> np.ndarray:
        pass
