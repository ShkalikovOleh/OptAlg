from ..optimizer import OptimizerWithHistory
import numpy as np


def decode(bin_code, value_range):
    dec_code = 0
    l = len(bin_code)
    i = l-1
    while i > 0:
        dec_code += bin_code[-i]*2**i
        i -= 1
    real = dec_code * (value_range[1] - value_range[0]) / (2 ** l - 1) + value_range[0]
    return real


def generate(length):
    return np.random.randint(2, size=length)


# antibody's coordinates are binary strings of length 22 representing coded real values
class Antibody:
    def __init__(self, id):
        self._id = id         # used to identify clones' parents,
        # maybe i will come up with a better solution later
        self._affinity = 0
        self._x1_bin = generate(22)
        self._x2_bin = generate(22)

    def compute_affinity(self, f, x1_range, x2_range):
        x1_dec = decode(self._x1_bin, x1_range)
        x2_dec = decode(self._x2_bin, x2_range)
        self._affinity = f(x1_dec, x2_dec)
        pass

    # temporal mutation method with fixed mutating bit
    # mutation rate should depend on affinity
    def mutate(self, mutation_probability):
        if np.random.random() < mutation_probability:
            self._x1_bin[6] = int(not self._x1_bin[6])
        if np.random.random() < mutation_probability:
            self._x2_bin[6] = int(not self._x2_bin[6])
        pass

    # not pythonic though, should read more about getters in python
    def get_affinity(self):
        return self._affinity

    def get_id(self):
        return self._id


# should consider reading more about other immune algorithms to create base AIS class
# for functions of two variables
class Clonalg(OptimizerWithHistory):
    def __init__(self, population_size, clone_multiplier,
                 max_mutation_rate):
        super().__init__()
        self._population_size = population_size
        self._population = []
        self._temp_population = []
        self._clone_multiplier = clone_multiplier
        self._max_mutation_rate = max_mutation_rate
        i = 0
        while i < population_size:
            self._population.append(Antibody(i))
            i += 1
        pass

    def _affinity(self, population, f, x1_range, x2_range):
        for ab in population:
            ab.compute_affinity(f, x1_range, x2_range)

    def _clone(self):
        for ab in self._population:
            self._temp_population.append([ab]*self._clone_multiplier)

    def _mutate(self):
        for j, ab in enumerate(self._temp_population):
            ab[1].mutate(self._max_mutation_rate*j/self._population_size)

    # comparing mutants to their parents and replacing
    def _replace(self):
        for ab in self._temp_population:
            parent = ab.get_id()
            if ab.get_affinity() > self._population[parent].get_affinity():
                self._population[parent] = ab
            pass

    # replacing d antibodies with low affinity with new generated antibodies
    def _edit(self):
        pass

    def optimize(self, f):
        pass

    # x1_range/x2_range - a tuple of two elements containing limits of a region in format (x_min, x_max)
    # wil move these parameters to init and implement optimize(f) method later
    # def clonalg(self, f, x1_range, x2_range, n_generations, maximize=True):
    #     while n_generations > 0:
    #         self._affinity(f, self._population, x1_range, x2_range)
    #
    #         self._clone()
    #
    #         sorted(self._temp_population, key=lambda ab: ab.get_affinity())
    #
    #         self._mutate()
    #
    #         self._affinity(f, self._temp_population, x1_range, x2_range)
    #
    #         self._replace()
    #
    #         sorted(self._population, key=lambda ab: ab.get_affinity())
    #
    #         self._edit()
    #
    #         n_generations -= 1
