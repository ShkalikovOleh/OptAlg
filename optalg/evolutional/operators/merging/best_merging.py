import numpy as np
from typing import Callable
from .merging_base import MergingOperator
from ..decoder import Decoder


class BestMerging(MergingOperator):

    def __call__(self, parents: np.ndarray,
                 children: np.ndarray, f: Callable, decoder: Decoder) -> np.ndarray:
        all = np.concatenate([parents, children], axis=0)
        phenotypes = decoder(all)
        f_values = np.apply_along_axis(f, axis=1, arr=phenotypes)
        best_idx = np.argsort(f_values)[:parents.shape[0]]
        return all[best_idx, ...]
