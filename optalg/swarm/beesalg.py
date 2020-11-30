from ..optimizer import Optimizer, OptimizeResult
from typing import Callable
import numpy as np
from numpy.random import uniform


class BeeColony(Optimizer):
    def __init__(self, n_variables, x_range, n_iter, delta, n_scout=30, n_research=25, n_elite=10,
                 bees_elite=5, bees_rest=3, maximize=False):
        super().__init__()
        self.n_variables = n_variables
        self.x_range = x_range
        self.n_scout = n_scout
        self.n_iter = n_iter
        self.n_research = n_research
        self.n_elite = n_elite
        self.delta = delta
        self.bees_elite = bees_elite
        self.bees_rest = bees_rest
        self.maximize = maximize
        self._solutions = []

    def random_neighbour(self, x0):
        return uniform(low=x0-self.delta, high=x0+self.delta,
                               size=self.n_variables)

    def optimize(self, f: Callable) -> OptimizeResult:
        self._solutions = list(uniform(low=self.x_range[:, 0], high=self.x_range[:, 1],
                               size=[self.n_scout, self.n_variables]))
        n = 1
        history = []
        while n <= self.n_iter:
            self._solutions.sort(key=f, reverse=self.maximize)
            history.append(np.array(self._solutions))
            for i in range(self.n_research):
                if i < self.n_elite:
                    r = self.bees_elite
                else:
                    r = self.bees_rest
                x0 = self._solutions[i]
                for j in range(r):
                    x = self.random_neighbour(self._solutions[i])
                    if (f(x) > f(x0)) == self.maximize:
                        x0 = x
                self._solutions[i] = x0

            for i in range(self.n_scout - self.n_research):
                self._solutions[-i-1] = uniform(low=self.x_range[:, 0], high=self.x_range[:, 1],
                                                size=self.n_variables)
            n += 1
        self._solutions.sort(key=f, reverse=self.maximize)
        history.append(np.array(self._solutions))
        xmin = self._solutions[0]
        res = OptimizeResult(f=f, x=xmin,
                             n_iter=len(history) - 1,
                             x_history=np.array(history))
        return res
