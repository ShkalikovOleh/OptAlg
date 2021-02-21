from typing import Callable, Generator, Union


def r_generator(r0: Union[int, float], multiplier: Union[int, float]):
    assert r0 > 0
    assert multiplier > 0

    def generator():
        i = 0
        while True:
            yield r0 * multiplier**i
            i += 1
    return generator


def check_sequence_increase(generator: Generator[float, None, None]):
    gen = generator()
    if next(gen) < next(gen):
        return True


def check_function_increase(func: Callable, a=10**-3, b=10**-2):
    assert a < b

    if func(a) < func(b):
        return True
