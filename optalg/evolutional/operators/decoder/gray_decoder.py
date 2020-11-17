import numpy as np
from typing import Union, Tuple
from .decoder_base import Decoder


class GrayDecoder(Decoder):

    def __init__(self, bounds: Tuple[Union[float, int], Union[float, int]]) -> None:
        self.__bounds = bounds

    def __call__(self, genotype: np.ndarray) -> np.ndarray:
        l = genotype.shape[1]

        def to_dec(bin_code):
            dec_code = 0.0
            i = l-1
            while i >= 0:
                dec_code += bin_code[-i]*2**i
                i -= 1
            real = dec_code * (self.__bounds[1] - self.__bounds[0]
                               ) / (2 ** l - 1) + self.__bounds[0]
            return real

        return np.apply_along_axis(to_dec, axis=1, arr=genotype)
