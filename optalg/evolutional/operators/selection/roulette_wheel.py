import numpy as np
from typing import Callable
from .selector_base import Selector


class RouletteWheel(Selector):

    def __call__(self, f: Callable, population: np.ndarray) -> np.ndarray:
        f_value = np.apply_along_axis(f, 1, population)
        probas = (f_value / np.sum(f_value)).reshape(-1,)

        idx = np.random.choice(
            np.arange(population.shape[0]), size=population.shape[0], p=probas)
        return idx
