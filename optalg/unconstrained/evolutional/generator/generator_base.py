import numpy as np
from abc import ABC, abstractmethod

from numpy.lib.type_check import real


class Generator(ABC):

    def __init__(self, n_genes: int) -> None:
        if n_genes <= 0:
            raise ValueError("Number of genes must be grater than 0")

        self._n_genes = n_genes

    @abstractmethod
    def __call__(self, population_size: int, n_variables: int) -> np.ndarray:
        """
        Generate population
        :param population_size: number of individuals in population
        :param n_variables: number of variables
        :param n_genes: number of genes per variables
        :return: new population with shape (population_size x n_variables x n_genes)
        :rtype: ndarray
        """
        pass
