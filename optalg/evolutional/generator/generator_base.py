import numpy as np
from abc import ABC, abstractmethod


class Generator(ABC):

    @abstractmethod
    def __call__(self, population_size: int, n_variables: int, n_genes: int) -> np.ndarray:
        """
        Generate population
        :param population_size: number of individuals in population
        :param n_variables: number of variables
        :param n_genes: number of genes per variables
        :return: new population with shape (population_size x n_variables x n_genes)
        :rtype: ndarray
        """
        pass
