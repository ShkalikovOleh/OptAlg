import numpy as np
from .generator_base import Generator


class BinaryGenerator(Generator):

    def __init__(self, n_genes: int = 22) -> None:
        super().__init__(n_genes)

    def __call__(self, population_size: int, n_variables: int) -> np.ndarray:
        shape = (population_size, n_variables, self._n_genes)
        size = population_size * n_variables * self._n_genes
        return np.random.randint(2, size=size).reshape(shape)
