import numpy as np
from .crossover_base import CrossoverOperator


class PointCrossover(CrossoverOperator):

    def __init__(self, k: int, proba) -> None:
        assert k > 0
        assert (proba > 0) and (proba < 1)

        self.__k = k
        self.__proba = proba

    def __call__(self, mating_genotypes: np.ndarray) -> np.ndarray:
        offspring_pop = np.copy(mating_genotypes)
        for i in range(mating_genotypes.shape[0], 2):
            for l in range(mating_genotypes.shape[2]):
                if self.__proba >= np.random.random():
                    points = np.random.choice(
                        np.arange(mating_genotypes.shape[1]), size=self.__k, replace=False)
                    points = np.sort(points)

                    begin = 0
                    for j in range(self.__k):
                        if j % 2 == 0:
                            offspring_pop[i, begin: points[j], l
                                          ] = mating_genotypes[i, begin:points[j], l]
                            offspring_pop[i + 1, begin: points[j], l
                                          ] = mating_genotypes[i+1, begin:points[j], l]
                        else:
                            offspring_pop[i, begin: points[j], l
                                          ] = mating_genotypes[i+1, begin:points[j], l]
                            offspring_pop[i + 1, begin: points[j], l
                                          ] = mating_genotypes[i, begin:points[j], l]

        return offspring_pop
