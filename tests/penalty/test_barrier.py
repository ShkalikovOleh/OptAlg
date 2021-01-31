import unittest
import numpy as np
from optalg.step import ArmijoBacktracking
from optalg.descent import BFGS, HookeJeeves
from optalg.penalty import Barrier
from optalg.stop_criteria import ArgumentNormCriterion
from ..inrange_assertion import InRangeAssertion


class BarrierTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + x[1]**2

    def test_convergence(self):
        step_opt = ArmijoBacktracking(1, 0.5)
        arg_criterion = ArgumentNormCriterion(10**-7)
        unc_opt = HookeJeeves(arg_criterion, step_opt,
                              HookeJeeves.generate_pertubation_vector(2, 0.1))

        opt = Barrier(unc_opt, epsilon=10**-4)

        res = opt.optimize(self.f, np.array([-1, -1]),
                           eq_constraints=[lambda x: x[0]-1],
                           ineq_constraints=[lambda x: x[0]+x[1]-2])

        self.assertInRange(res.x, np.array([1, 0]), 10**-3)

    def test_get_history(self):
        step_opt = ArmijoBacktracking(1, 0.5)
        arg_criterion = ArgumentNormCriterion(10**-7)
        unc_opt = HookeJeeves(arg_criterion, step_opt,
                              HookeJeeves.generate_pertubation_vector(2, 0.1))

        opt = Barrier(unc_opt, epsilon=10**-4)

        res = opt.optimize(self.f, np.array([-1, -1]),
                           eq_constraints=[lambda x: x[0]-1],
                           ineq_constraints=[lambda x: x[0]+x[1]-2])

        self.assertEqual(2, res.x_history.ndim)
