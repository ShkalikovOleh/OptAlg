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
from optalg.step import StepDivision
from optalg.descent import GradientDescent
from optalg.stop_criteria import GradientNormCriterion

def f(x):
  return x[0]**2 + np.exp(x[1]**2)

gnCriterion = GradientNormCriterion(10**-3)
step_opt = StepDivision(1, 0.5)

optimizer = GradientDescentStepDecrease(np.array([[-3],[4]]), gnCriterion, step_opt)
xmin = optimizer.optimize(f)
```

## Available algorithms

### Descent

#### Gradient
Methods based on descent to minimum by gradient-like direction.

- [Gradient descent with step decrease](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gd_step_decrease.py) - gradient descent with step dividing if the function value at the new point is greater than the function value at the previous point.

- [Gradient descent fastest](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gd_fastest.py) - gradient descent with step determine by other optimizer.

- [Cojugate gradients descent](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gd_conjugate.py) - descent direction is the sum of gradient in current point and the weighted direction from the previous iteration.
Avaliable variations:
  - Fletcher-Reeves
  - Polak-Ribiere
  - Hestenes-Stiefel
  - Dai–Yuan

#### Newton
Second-order descent algorithms

- [Newton](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/newton/newton.py) - descent direction is dot product of the hessian and gradient.

On each step descent direction multiplies by step size.
Avaliable descent's *step size* calculation methods:

- [SimpleSearch](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/iterative/simple_search.py) - uniform selection of n values from the interval.

- [Fibonacci](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/iterative/fibonacci.py) - 1-dimensional optimisation for **unimodal** functions(in our case argument is step size). Consequently converges search region until diameter < epsilon; x_min is center of resulting region.


### Immune
Artificial immune system

- [ClonAlg](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/immune/clonalg.py) - clonal selection algorithm
