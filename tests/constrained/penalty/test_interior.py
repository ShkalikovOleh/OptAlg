import unittest
import numpy as np
from optalg.line_search import ArmijoBacktracking
from optalg.unconstrained import HookeJeeves
from optalg.constrained.penalty import Interior
from optalg.stop_criteria import IterationNumberCriterion
from ...inrange_assertion import InRangeAssertion


class InteriorTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + x[1]**2

    def test_convergence(self):
        step_opt = ArmijoBacktracking(1, 0.5)
        arg_criterion = IterationNumberCriterion(500)
        unc_opt = HookeJeeves(arg_criterion,
                              HookeJeeves.generate_pertubation_vector(2, 0.1))

        opt = Interior(unc_opt, epsilon=10**-4)

        res = opt.optimize(self.f, np.array([-1, -1]),
                           eq_constraints=[lambda x: x[0]-1],
                           ineq_constraints=[lambda x: x[0]+x[1]-2])

        self.assertInRange(res.x, np.array([1, 0]), 10**-3)
