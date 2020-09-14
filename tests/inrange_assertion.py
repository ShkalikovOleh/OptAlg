import numpy as np


class InRangeAssertion():
    def assertInRange(self, x_opt, xmin, delta):
        if np.linalg.norm(x_opt - xmin) > delta:
            raise AssertionError("Result of out of expected range")
