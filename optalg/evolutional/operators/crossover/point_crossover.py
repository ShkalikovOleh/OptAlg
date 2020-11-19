import numpy as np
from numpy.random import choice, random
from .crossover_base import CrossoverOperator


class PointCrossover(CrossoverOperator):

    def __init__(self, k: int, proba) -> None:
        assert k > 0
        assert (proba > 0) and (proba < 1)

        self.__k = k
        self.__proba = proba

    def __call__(self, population: np.ndarray, parents_idx: np.ndarray) -> np.ndarray:
        assert parents_idx.shape[1] == 2

        children = np.copy(population)
        _, n_var, n_bin = population.shape
        for i in range(parents_idx.shape[0]):
            p1 = parents_idx[i, 0]
            p2 = parents_idx[i, 1]

            for j in range(n_var):
                t = 0
                if self.__proba >= random(): #check here more productive
                    points = choice(range(1, n_bin-1),
                                    size=self.__k, replace=False)
                    points = np.sort(points)

                    for l, point in enumerate(points):
                        if l % 2 == 0:
                            children[p1, j, t:point] = population[p2, j, t:point]
                            children[p2, j, t:point] = population[p1, j, t:point]
                        t = point

                    if self.__k % 2 == 0:
                        children[p1, j, t:] = population[p2, j, t:]
                        children[p1, j, t:] = population[p2, j, t:]

        return children
