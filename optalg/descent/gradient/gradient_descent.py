from ...optimizer import OptimizerWithHistory


class GradientDescentOptimizer(OptimizerWithHistory):

    def __init__(self, x0, stop_criteria):
        super().__init__()
        self._x0 = x0
        self._stop_criteria = stop_criteria

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        if value.shape == self._x0.shape:
            self._x0 = value
