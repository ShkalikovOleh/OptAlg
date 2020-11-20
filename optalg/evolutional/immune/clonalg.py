import numpy as np
from typing import Callable
from ...stop_criteria import StopCriterion
from ..evolutional_base import EvolutionalBase
from ..generator import BinaryGenerator
from ..decoder import GrayDecoder


class ClonAlg(EvolutionalBase):

    def __init__(self, n_variables: int, population_size: int,
                 stop_criterion: StopCriterion, range, clone_multiplier: int = 5,
                 max_mutation_rate: float = 0.3, to_replace: int = 2) -> None:

        assert clone_multiplier > 0
        assert max_mutation_rate <= 1 and max_mutation_rate > 0
        assert to_replace >= 0

        super().__init__(n_variables, population_size, stop_criterion,
                         BinaryGenerator(22), GrayDecoder(range))

        self.__clone_mult = clone_multiplier
        self.__max_mut_rate = max_mutation_rate
        self.__to_replace = to_replace

    def _select(self, f: Callable, population: np.ndarray):
        pop_size, n_var, n_genes = population.shape
        k = self.__clone_mult

        sorted_idx = np.argsort(np.apply_along_axis(
            f, axis=1, arr=self._decoder(population)))
        population = population[sorted_idx]

        mating_pool = np.empty((k*pop_size, n_var, n_genes))
        for i in range(0, pop_size, k):  # clone
            mating_pool[i:i+k] = population[i]

        return mating_pool

    def __mutate(self, individual, proba):
        for i in range(individual.shape[0]):
            mask = np.random.choice(
                [True, False], size=individual.shape[1], p=[proba, 1 - proba])
            individual[i, mask] = 1 - individual[i, mask]
        return individual

    def _reproduce(self, f: Callable, mating_pool: np.ndarray):
        children = np.empty(
            (self._population_size, self._n_variables, mating_pool.shape[2]))

        for j in range(0, mating_pool.shape[0], self.__clone_mult):
            mut_rate = (j // self.__clone_mult + 1) * \
                self.__max_mut_rate / self._population_size

            mutated = np.empty((self.__clone_mult,
                                self._n_variables, mating_pool.shape[2]))
            for i in range(self.__clone_mult):
                mutated[i] = self.__mutate(mating_pool[j+i], mut_rate)

            min_idx = np.argmin(np.apply_along_axis(
                f, axis=1, arr=self._decoder(mutated)))
            children[j // self.__clone_mult] = mutated[min_idx]

        return children

    def _replace(self, f: Callable, children: np.ndarray, parents: np.ndarray):
        children_phenotypes = self._decoder(children)
        parents_phenotypes = self._decoder(parents)

        f_children = np.apply_along_axis(f, axis=1, arr=children_phenotypes)
        f_parents = np.apply_along_axis(f, axis=1, arr=parents_phenotypes)

        mask = f_children > f_parents
        children[mask] = parents[mask]  # insert
        f_children[mask] = f_parents[mask]

        worst_idx = np.argsort(f_children)[-self.__to_replace:]  # edit
        children[worst_idx] = self._generator(2, self._n_variables)

        return children
