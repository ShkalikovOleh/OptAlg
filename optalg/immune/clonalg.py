from ..optimizer import OptimizerWithHistory
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
    real = dec_code * (value_range[1] - value_range[0]) / (2 ** l - 1) + value_range[0]
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
        return np.array(x_dec).reshape((self.__n_variables ,1))

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

    @affinity.setter
    def affinity(self, aff):
        self.__affinity = aff


class ClonAlg(OptimizerWithHistory):
    def __init__(self, n_variables, x_range, population_size=10, n_generations=30, clone_multiplier=5,
                 max_mutation_rate=0.3, to_replace=2):
        super().__init__()
        self._population = []
        self._temp_population = []
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
            for i in range(self.clone_multiplier):
                self._temp_population.append(deepcopy(ab))

    def _mutate(self):
        l = self.population_size*self.clone_multiplier
        for j in range(l):
            self._temp_population[j].mutate(self.max_mutation_rate * (j + 1) / l)

    # comparing mutants to their parents and replacing
    def _insert(self):
        for ab in self._population:
            for clone in self._temp_population:
                if ab.id == clone.id and ab.affinity > clone.affinity:
                    ab.x_bin = clone.x_bin
                    ab.affinity = clone.affinity

    # replacing d antibodies with low affinity with new generated antibodies
    def _edit(self, f):
        for i in range(self.to_replace):
            self._population[-(i+1)] = Antibody(self.n_variables, self.x_range, id=self._population[-(i+1)].id)
            self._population[-(i+1)].compute_affinity(f)

    @property
    def history(self):
        return self._history

    def optimize(self, f):
        self.history_reset()
        self._population = [Antibody(self.n_variables, self.x_range, id=i) for i in range(self.population_size)]
        self._affinity(self._population, f)
        self._population = sorted(self._population, key=lambda ab: ab.affinity)
        self._history.append([ab.get_coordinates() for ab in self._population])

        g = self.n_generations
        while g > 0:
            self._temp_population = []
            self._clone()
            self._mutate()
            self._affinity(self._temp_population, f)
            self._insert()
            self._population = sorted(self._population, key=lambda ab: ab.affinity)
            self._edit(f)
            self._population = sorted(self._population, key=lambda ab: ab.affinity)
            self._history.append([ab.get_coordinates() for ab in self._population])
            g -= 1

        return self._population[0].get_coordinates()
