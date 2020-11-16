import numpy as np
from typing import Callable
from ..stop_criteria import StopCriterion
from ..optimizer import Optimizer, OptimizeResult


class Genetic(Optimizer):

    def __init__(self, stop_criterion: StopCriterion,
                 n_variables: int,
                 n_population: int,
                 decoder,
                 selector,
                 crossover,
                 mutator,
                 bin_length: int = 22) -> None:
        assert(n_variables > 0)
        assert(n_population > 0)
        assert(bin_length > 0)

        super().__init__()
        self.__stop_criterion = stop_criterion
        self.__selector = selector
        self.__decoder = decoder
        self.__crossover = crossover
        self.__mutator = mutator
        self.__n_variables = n_variables
        self.__bin_lenght = bin_length
        self.__n_population = n_population
        self.__history = []

    def __generate_population(self):
        size = self.__n_population * self.__bin_lenght * self.__n_variables
        population = np.random.randint(2, size=size). \
            reshape(self.__n_population, self.__bin_lenght, self.__n_variables)
        return population

    def optimize(self, f: Callable) -> OptimizeResult:
        self.__history.clear()

        population = self.__generate_population()
        self.__history.append(self.__decoder(population))

        while not self.__stop_criterion.match(f, self.__history):
            mating_pool_idx = self.__selector(f, self.__decoder(population))
            mating_pool = np.empty_like(population)
            for i, idx in enumerate(mating_pool_idx):
                mating_pool[i, :, :] = population[idx, :, :]

            offspring_genotypes = self.__crossover(mating_pool)
            population = self.__mutator(offspring_genotypes)

            self.__history.append(self.__decoder(population))

        idx = np.argmax(np.apply_along_axis(f, axis=1, arr=self.__history[-1]))
        x = self.__history[-1][idx, :]
        res = OptimizeResult(f=f, x=x, x_history=np.array(self.__history))

        self.__history.clear()

        return res
