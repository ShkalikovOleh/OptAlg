import unittest
import numpy as np
from optalg.unconstrained import HookeJeeves
from optalg.stop_criteria import IterationNumberCriterion
from optalg.collector import SaveAllCollector
from ..inrange_assertion import InRangeAssertion

class HookeJeevesTests(unittest.TestCase, InRangeAssertion):

    @staticmethod
    def f(x):
        return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

    def test_convergence(self):
        criterion = IterationNumberCriterion(5*10**3)

        p_vec = HookeJeeves.generate_pertubation_vector(2, 0.5)
        opt = HookeJeeves(criterion, p_vec)
        res = opt.optimize(self.f, np.array([0, 0]))

        self.assertInRange(res.x, np.array([3, 2]), 10**-2)
