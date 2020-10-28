from ..optimizer import OptimizeResult, Optimizer
import numpy as np
from copy import deepcopy


# value_range is a tuple of 2 elements
def decode(bin_code, value_range):
    dec_code = 0.0
    l = len(bin_code)
    i = l-1
    while i > 0:
        dec_code += bin_code[-i]*2**i
        i -= 1
    real = dec_code * (value_range[1] - value_range[0]
                       ) / (2 ** l - 1) + value_range[0]
    return real


def generate(length):
    return np.random.randint(2, size=length)


# antibody's coordinates are binary strings of length 22 representing coded real values
class Antibody:
    def __init__(self, n_variables, x_range, id=0):
        self.__id = id
        self.__n_variables = n_variables
        self.__affinity = 0
        self.__x_range = x_range
        self.__x_bin = np.array([generate(22) for i in range(n_variables)])

    def get_coordinates(self):
        x_dec = []
        for i in range(len(self.__x_bin)):
            x_dec.append(decode(self.__x_bin[i], self.__x_range[i]))
        return np.array(x_dec)

    def compute_affinity(self, f):
        x_dec = self.get_coordinates()
        self.__affinity = f(x_dec)

    def mutate(self, mutation_probability):
        mask = [np.random.choice([True, False], size=self.__x_bin.shape[1],
                                 p=[mutation_probability, 1 - mutation_probability])
                for i in range(self.__x_bin.shape[0])]
        for i in range(self.__x_bin.shape[0]):
            self.__x_bin[i, mask[i]] = 1 - self.__x_bin[i, mask[i]]

    @property
    def affinity(self):
        return self.__affinity

    @property
    def id(self):
        return self.__id

    @property
    def x_bin(self):
        return self.__x_bin

    @x_bin.setter
    def x_bin(self, x_bin):
        self.__x_bin = x_bin


class ClonAlg(Optimizer):
    def __init__(self, n_variables, x_range, population_size=10, n_generations=30, clone_multiplier=5,
                 max_mutation_rate=0.3, to_replace=2):
        super().__init__()
        self.clone_multiplier = clone_multiplier
        self.max_mutation_rate = max_mutation_rate
        self.to_replace = to_replace
        self.x_range = x_range
        self.population_size = population_size
        self.n_variables = n_variables
        self.n_generations = n_generations

    @staticmethod
    def _affinity(population, f):
        for ab in population:
            ab.compute_affinity(f)

    def _clone(self):
        for ab in self._population:
            clones = [deepcopy(ab) for _ in range(self.clone_multiplier)]
            self._temp_population[ab.id] = clones

    def _mutate(self):
        for i, ab in enumerate(self._population):
            for clone in self._temp_population[ab.id]:
                clone.mutate(self.max_mutation_rate *
                             (i+1) / self.population_size)

    # comparing mutants to their parents and replacing
    def _insert(self, f):
        for ab in self._population:
            clones = self._temp_population[ab.id]
            min_idx = np.argmin([clone.affinity for clone in clones])
            if ab.affinity > clones[min_idx].affinity:
                ab.x_bin = clones[min_idx].x_bin
                ab.compute_affinity(f)

    # replacing d antibodies with low affinity with new generated antibodies
    def _edit(self, f):
        for i in range(self.to_replace):
            self._population[-(i+1)] = Antibody(self.n_variables,
                                                self.x_range, id=self._population[-(i+1)].id)
            self._population[-(i+1)].compute_affinity(f)

    def optimize(self, f):
        self._population = [Antibody(self.n_variables, self.x_range, id=i)
                            for i in range(self.population_size)]
        self._affinity(self._population, f)
        self._population = sorted(self._population, key=lambda ab: ab.affinity)

        self._history = []
        self._history.append([ab.get_coordinates() for ab in self._population])

        g = self.n_generations
        while g > 0:
            self._temp_population = {}
            self._clone()
            self._mutate()
            clones = [clone for clones in self._temp_population.values()
                      for clone in clones]
            self._affinity(clones, f)
            self._insert(f)
            self._population = sorted(
                self._population, key=lambda ab: ab.affinity)
            self._edit(f)
            self._population = sorted(
                self._population, key=lambda ab: ab.affinity)
            self._history.append([ab.get_coordinates()
                                  for ab in self._population])
            g -= 1

        x = self._population[0].get_coordinates()
        xhist = np.array(self._history)
        res = OptimizeResult(f=f, x=x, x_history=xhist)

        return res
