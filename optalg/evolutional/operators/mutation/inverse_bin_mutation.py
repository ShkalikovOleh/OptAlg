import numpy as np
from numpy.random import choice, random
from .mutation_base import MutationOperator


class InverseBinaryMutation(MutationOperator):

    def __init__(self, mut_proba: float, k: int = 1) -> None:
        assert k > 0
        assert (mut_proba >= 0 and mut_proba < 1)
        self.__mut_proba = mut_proba
        self.__k = k

    def __call__(self, pop_genotypes) -> np.ndarray:
        n_genes = pop_genotypes.shape[2]
        for i in range(pop_genotypes.shape[0]):
            for l in range(pop_genotypes.shape[1]):
                if random() > 1 - self.__mut_proba:
                    j = choice(np.arange(n_genes), size=self.__k, replace=False)
                    pop_genotypes[i, l, j] = 1 - pop_genotypes[i, l, j]

        return pop_genotypes


