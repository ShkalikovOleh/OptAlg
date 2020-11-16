import numpy as np
from abc import ABC, abstractmethod


class Decoder(ABC):

    @abstractmethod
    def __call__(self, genotype: np.ndarray) -> np.ndarray:
        pass
