import numpy as np
from .decoder_base import Decoder


class GrayDecoder(Decoder):

    def __init__(self, bounds) -> None:
        if isinstance(bounds, tuple):
            self.__bounds = [bounds]
        else:
            self.__bounds = bounds

    def __call__(self, population: np.ndarray) -> np.ndarray:
        pop_size, n_var, n_genes = population.shape
        result = np.empty((pop_size, n_var))

        def bin_to_dec(bin):
            dec = 0
            for i in range(n_genes):
                dec += bin[i]*2**(n_genes-i-1)
            return dec

        for j in range(n_var):
            dec = np.apply_along_axis(
                bin_to_dec, axis=1, arr=population[:, j, :])

            bound = self.__bounds[j % len(self.__bounds)]
            result[:, j] = dec * (bound[1] - bound[0]) / \
                (2**n_genes - 1) + bound[0]

        return result
