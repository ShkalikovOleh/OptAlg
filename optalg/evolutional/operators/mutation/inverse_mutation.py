import numpy as np
from numpy.core.defchararray import replace
from .mutation_base import MutationOperator


class InverseMutation(MutationOperator):

    def __init__(self, mut_proba: float, k: int = 1) -> None:
        assert k > 0
        assert (mut_proba >= 0 and mut_proba < 1)
        self.__mut_proba = mut_proba
        self.__k = k

    def __call__(self, pop_genotypes) -> np.ndarray:
        n_genes = pop_genotypes.shape[1]
        for i in range(pop_genotypes.shape[0]):
            for l in range(pop_genotypes.shape[2]):
                if np.random.random() > 1 - self.__mut_proba:
                    j = np.random.choice(np.arange(n_genes), size=self.__k, replace=False)
                    pop_genotypes[i, j, l] = 1 - pop_genotypes[i, j, l]

        return pop_genotypes


