import unittest
import numpy as np
from autograd.numpy import sqrt
from optalg.line_search import ArmijoBacktracking
from optalg.unconstrained.descent import BFGS
from optalg.constrained.penalty import Penalty
from optalg.stop_criteria import GradientNormCriterion
from ...inrange_assertion import InRangeAssertion


class PenaltyTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return sqrt(x[0]**2 + x[1]**2)

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = ArmijoBacktracking(1, 0.5)
        unc_opt = BFGS(gnCriterion, step_opt)

        opt = Penalty(unc_opt)

        res = opt.optimize(self.f, np.array([-6, 9]),
                           eq_constraints=[lambda x: x[0]+x[1]-2],
                           ineq_constraints=[lambda x: -x[0]])

        self.assertInRange(res.x, np.array([1, 1]), 10**-3)

    def test_get_history(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = ArmijoBacktracking(1, 0.5)
        unc_opt = BFGS(gnCriterion, step_opt)

        opt = Penalty(unc_opt)
        res = opt.optimize(self.f, np.array([-4, 6]),
                           eq_constraints=[lambda x: x[0]+x[1]-2],
                           ineq_constraints=[lambda x: -x[0]])

        self.assertEqual(2, res.x_history.ndim)
