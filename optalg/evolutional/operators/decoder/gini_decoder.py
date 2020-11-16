from typing import Tuple
import numpy as np
from .decoder_base import Decoder


class GiniDecoder(Decoder):

    def __init__(self, bounds: Tuple, l: int) -> None:
        self.__bounds = bounds
        self.__l = l

    def __call__(self, genotype: np.ndarray) -> np.ndarray:
        assert(genotype.shape[1] == self.__l)

        def to_dec(bin_code):
            dec_code = 0.0
            i = self.__l-1
            while i >= 0:
                dec_code += bin_code[-i]*2**i
                i -= 1
            real = dec_code * (self.__bounds[1] - self.__bounds[0]
                               ) / (2 ** self.__l - 1) + self.__bounds[0]
            return real
        return np.apply_along_axis(to_dec, axis=1, arr=genotype)