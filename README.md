# OptAlg
Set of optimization algorithms.

## Usage
```python
from optalg.<subpackage> import <algo-name>

def f(x):
  return <your-function value>(where argument x_i = x[i])

optimizer = <algo-name>(params...)
res = optimizer.optimize(f, [init_state]) #optimization result
```

For methods **requiring gradient and hessian calculations, use** `autograd.numpy` instead of `numpy`
for define objective function.
For example:

```python
import numpy as np
from autograd.numpy import sin
from optalg.step import ArmijoBacktracking
from optalg.descent import GradientDescent
from optalg.stop_criteria import GradientNormCriterion


def f(x):
  return x[0]**2 + sin(x[1]**2)


gnCriterion = GradientNormCriterion(10**-3)
step_opt = ArmijoBacktracking(1, 0.5)
optimizer = GradientDescent(gnCriterion, step_opt)

res = optimizer.optimize(f, np.array([-3, 1]))
res.x #optimum
```

## Available algorithms

### Descent
Methods based on descent to the optimum by something direction.

On each step descent direction multiplies by step size.
Avaliable descent's *step size* calculation methods:

- [Fixed](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/step/fixed.py) - constant step size

- [GridSearch](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/step/grid_search.py) - uniform selection of n values from the interval.

- [ArmijoBacktracking](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/step/armijo_backtracking.py) - step dividing if the function value at the new point does not satisfy armijo condition.

- [BisectionWolfe](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/step/bisection_wolfe.py) - bisection method that either computes a step size satisfying the weak Wolfe conditions or sends the function values to -inf.

- [Fibonacci](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/step/fibonacci.py) - 1-dimensional optimisation for **unimodal** functions(in our case argument is step size). Consequently converges search region until diameter < epsilon; x_min is center of resulting region.

#### Search
Methods that does not require differentiability of the objective function.
For direction calculation uses another search methods.

- [Hooke-Jeeves](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/search/hooke_jeeves.py) - pattern search. Descent direction is the best combination of coordinates of the pertubation vector.


#### Gradient
Methods based on descent to minimum by gradient-like direction.

- [Gradient descent](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gradient_descent.py) - simple gradient descent

- [Cojugate gradients descent](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/gradient/gd_conjugate.py) - descent direction is the sum of gradient in current point and the weighted direction from the previous iteration.
Avaliable variations:
  - Fletcher-Reeves
  - Polak-Ribiere
  - Hestenes-Stiefel
  - Dai-Yuan

#### Newton
Second-order descent algorithms

- [Newton](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/newton/newton.py) - descent direction is dot product of the hessian and gradient.

- Quasi Newton - inverse hessian replaces with an approximate value
  - [BFGS](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/newton/BFGS.py) - Broyden-Fletcher-Goldfarb-Shanno
  - [DFP](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/newton/DFP.py) - Davidon-Fletcher-Powell
  - [SR1](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/newton/SR1.py) - Symmetric Rank 1 method
  - [Broyden](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/descent/newton/broyden.py) - Broyden's method

### Immune
Artificial immune system

- [ClonAlg](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/immune/clonalg.py) - clonal selection algorithm
