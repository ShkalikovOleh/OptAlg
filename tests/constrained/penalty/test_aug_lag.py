import unittest
import numpy as np
from optalg.line_search import ArmijoBacktracking
from optalg.unconstrained.descent import BFGS
from optalg.constrained.penalty import AugmentedLagrangian
from optalg.stop_criteria import GradientNormCriterion
from ...inrange_assertion import InRangeAssertion


class AugmentedLagrangianTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return x[0]**2 + x[1]**2

    def test_convergence(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = ArmijoBacktracking(1, 0.5)
        unc_opt = BFGS(gnCriterion, step_opt)

        opt = AugmentedLagrangian(unc_opt)

        res = opt.optimize(self.f, np.array([4, 8]),
                           eq_constraints=[lambda x: x[0]-1],
                           ineq_constraints=[lambda x: x[0]+x[1]-2])

        self.assertInRange(res.x, np.array([1, 0]), 10**-3)

    def test_get_history(self):
        gnCriterion = GradientNormCriterion(10**-3)
        step_opt = ArmijoBacktracking(1, 0.5)
        unc_opt = BFGS(gnCriterion, step_opt)

        opt = AugmentedLagrangian(unc_opt)
        res = opt.optimize(self.f, np.array([-4, 6]),
                           eq_constraints=[lambda x: x[0]+x[1]-2],
                           ineq_constraints=[lambda x: -x[0]])

        self.assertEqual(2, res.x_history.ndim)
