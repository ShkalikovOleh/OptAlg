import numpy as np
from abc import abstractmethod
from typing import Callable, List
from ...optimizer import Optimizer, OptimizeResult
from ...collector import CollectorBase, reset_collectors, accept_collectors
from ...stop_criteria import StopCriterion
from .generator import Generator
from .decoder import Decoder


class EvolutionalBase(Optimizer):

    def __init__(self, n_variables: int, population_size: int, stop_criterion: StopCriterion,
                 generator: Generator, decoder: Decoder,
                 population_collectors: List[CollectorBase]) -> None:
        assert n_variables > 0
        assert population_size > 0

        super().__init__()
        self._n_variables = n_variables
        self._population_size = population_size
        self._stop_criterion = stop_criterion
        self._generator = generator
        self._decoder = decoder
        self._population_collector = population_collectors

    @abstractmethod
    def _select(self, f: Callable, population: np.ndarray):
        pass

    @abstractmethod
    def _reproduce(self, f: Callable, mating_pool: np.ndarray):
        pass

    @abstractmethod
    def _replace(self, f: Callable, children: np.ndarray, parents: np.ndarray):
        pass

    def optimize(self, f: Callable) -> OptimizeResult:
        population = self._generator(self._population_size, self._n_variables)
        phenotypes = self._decoder(population)
        n_iter = 0

        reset_collectors(self._population_collector)
        accept_collectors(self._population_collector, population)

        self._stop_criterion.accept(f, population)

        while not self._stop_criterion.match():
            n_iter += 1

            mating_pool = self._select(f, population)
            children = self._reproduce(f, mating_pool)
            population = self._replace(f, children, population)
            phenotypes = self._decoder(population)

            self._stop_criterion.accept(f, phenotypes)
            accept_collectors(self._population_collector, phenotypes)

        idx = np.argmin(np.apply_along_axis(f, axis=1, arr=phenotypes))
        xmin = phenotypes[idx, :]
        res = OptimizeResult(f=f, x=xmin,
                             n_iter=n_iter)

        return res
