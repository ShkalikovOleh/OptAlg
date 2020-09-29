# OptAlg
Set of optimization algorithms.

## Usage
```python
from optalg.<subpackage> import <algo-name>

def f(x):
  return <your-function value>(where argument x_i = x[i])

optimizer = <algo-name>(params...)
xmin = optimizer.optimize(f)
```

For methods **requiring gradient and hessian calculations, use** `autograd.numpy` instead of `numpy`
to define objective function.
For example:

```python
import autograd.numpy as np
from optalg.descent import GradientDescentStepDecrease
from optalg.stop_criteria import GradientNormCriterion

def f(x):
  return np.sin(x[0])**2 + np.sin(x[1])**2

gnCriterion = GradientNormCriterion(10**-3)
optimizer = GradientDescentStepDecrease(np.array([[-3],[4]]), 1, 0.5, gnCriterion)
xmin = optimizer.optimize(f)
```

## Algorithms types

### Iterative
Methods based on iteration over values and calculating the function for each of them.

- [SimpleSearch](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/iterative/simple_search.py) - uniform selection of n values from the interval.

- [Fibonacci](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/iterative/fibonacci.py) - 1-dimensional optimisation for **unimodal** functions. Consequently converges search region until diameter < epsilon; x_min is center of resulting region.

### Descent

#### Gradient
Methods based on descent to minimum by gradient-like direction.

- [Gradient descent with step decrease](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gd_step_decrease.py) - gradient descent with step dividing if the function value at the new point is greater than the function value at the previous point.

- [Gradient descent fastest](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gd_fastest.py) - gradient descent with step determine by other optimizer.

- [Cojugate gradients descent](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gd_conjugate.py) - descent direction is the sum of gradient in current point and the weighted direction from the previous iteration.
Avaliable variations:
  - Fletcher–Reeves
  - Polak–Ribiere
  - Hestenes-Stiefel
  - Dai–Yuan
