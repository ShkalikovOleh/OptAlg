import numpy as np
from numpy.random import choice, permutation
from typing import Callable
from .selector_base import Selector


class RouletteWheel(Selector):

    def __init__(self, n_parents=2):
        super().__init__()
        self.__n_parents = n_parents

    def __call__(self, f: Callable, population: np.ndarray) -> np.ndarray:
        pop_size = population.shape[0]

        f_value = np.apply_along_axis(f, axis=1, arr=population).reshape(-1,)
        f_value = np.abs(np.max(f_value) - f_value +
                         np.min(f_value))  # for minimizing

        p = (f_value / np.sum(f_value))
        idx = choice(np.arange(pop_size), size=pop_size, p=p)  # mating pool

        parents_idx = [idx]
        for i in range(1, self.__n_parents):
            parents_idx.append(permutation(idx))

        return np.asarray(parents_idx).T
