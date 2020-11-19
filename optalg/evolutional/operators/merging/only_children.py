import numpy as np
from typing import Callable
from .merging_base import MergingOperator
from ..decoder import Decoder


class OnlyChildrenMerging(MergingOperator):

    def __call__(self, parents: np.ndarray,
                 children: np.ndarray, f: Callable, decoder: Decoder) -> np.ndarray:
        return children
