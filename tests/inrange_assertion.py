import numpy as np


class InRangeAssertion():
    def assertInRange(self, x_opt, xmin, delta):
        diff = np.linalg.norm(x_opt - xmin)
        if diff > delta:
            raise AssertionError(f"Result {x_opt} out of expected range. Difference: {diff}")
