import numpy as np
from .mutation_base import MutationOperator


class SimpleMutation(MutationOperator):

    def __init__(self, mut_proba: float) -> None:
        assert (mut_proba >= 0 and mut_proba < 1)
        self.__mut_proba = mut_proba

    def __call__(self, pop_genotypes) -> np.ndarray:
        n_genes = pop_genotypes.shape[1]
        for i in range(pop_genotypes.shape[0]):
            for k in range(pop_genotypes.shape[2]):
                if np.random.random() > 1 - self.__mut_proba:
                    j = np.random.randint(n_genes)
                    pop_genotypes[i, j, k] = 1 - pop_genotypes[i, j, k]

        return pop_genotypes


