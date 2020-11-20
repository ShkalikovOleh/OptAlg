import numpy as np
from .generator_base import Generator


class BinaryGenerator(Generator):

    def __call__(self, population_size: int, n_variables: int, n_genes: int) -> np.ndarray:
        shape = (population_size, n_variables, n_genes)
        size = population_size * n_variables, n_genes
        return np.random.randint(2, size=size).reshape(shape)
