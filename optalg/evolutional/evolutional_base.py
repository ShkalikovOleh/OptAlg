import numpy as np
from abc import abstractmethod
from typing import Callable
from ..optimizer import Optimizer, OptimizeResult
from ..stop_criteria import StopCriterion
from .generator import Generator
from .decoder import Decoder


class EvolutionalBase(Optimizer):

    def __init__(self, stop_criterion: StopCriterion, generator: Generator, decoder: Decoder) -> None:
        super.__init__()
        self._stop_criterion = stop_criterion
        self._generator = generator
        self._decoder = decoder

    @abstractmethod
    def _select(self, population: np.ndarray):
        pass

    @abstractmethod
    def _reproduct(self, mating_pool: np.ndarray):
        pass

    @abstractmethod
    def _replace(self, children: np.ndarray, parents: np.ndarray):
        pass

    def optimize(self, f: Callable) -> OptimizeResult:
        population = self._generator.generate()

        history = []
        history.append(self._decoder(population))

        while self._stop_criterion.match(f, history):
            mating_pool = self._select(population)
            children = self._reproduct(mating_pool)
            population = self._replace(children, population)
            history.append(self._decoder(population))

        idx = np.argmin(np.apply_along_axis(f, axis=1, arr=history[-1]))
        xmin = history[-1][idx, :]
        res = OptimizeResult(f=f, x=xmin,
                             n_iter=len(history) - 1,
                             x_history=np.array(history))

        return res
