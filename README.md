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
from optalg.line_search import ArmijoBacktracking
from optalg.unconstrained.descent import GradientDescent
from optalg.stop_criteria import GradientNormCriterion


def f(x):
  return x[0]**2 + sin(x[1]**2)


gnCriterion = GradientNormCriterion(10**-3)
step_opt = ArmijoBacktracking(1, 0.5)
optimizer = GradientDescent(gnCriterion, step_opt)

res = optimizer.optimize(f, np.array([-3, 1]))
res.x #optimum
```

## Unconstrained algorithms

### Gradient free
Methods that does not require differentiability of the objective function.
For direction calculation uses another search methods.

- Nelder-Mead - simplex reflection, contraction, expansion and shrink to the minimum of the objective function.
- Hooke-Jeeves - pattern search. Descent direction is the best combination of coordinates of the pertubation vector.

### Gradient
Methods based on descent to minimum by gradient-like direction.

- Gradient descent - simple gradient descent

- Cojugate gradients descent - descent direction is the sum of gradient in current point and the weighted direction from the previous iteration.
Avaliable variations:
  - Fletcher-Reeves
  - Polak-Ribiere
  - Hestenes-Stiefel
  - Dai-Yuan

### Newton
Second-order descent algorithms

- Newton - descent direction is dot product of the hessian and gradient.

- Quasi Newton - inverse hessian replaces with an approximate value
  - BFGS - Broyden-Fletcher-Goldfarb-Shanno
  - DFP - Davidon-Fletcher-Powell
  - SR1 - Symmetric Rank 1 method
  - Broyden - Broyden's method

### Evolutional
Methods based on natural evolution. On each iteration methods select "best"
individual from population, reproduce new generation and replace previous individuals.

#### Immune
Artificial immune system

- ClonAlg - clonal selection algorithm

## Constrained algorithms

### Penalty
Set of method for constrained optimization that use penalty function for representing constraints

- Penalty - penalty function for external point
- Interior - barrier function for interior point
- Augmented Lagragian - modified Lagrangian + inequality constraints

## Line search
On each step descent direction multiplies by step size.
Avaliable descent's *step size* calculation methods:

- FixedStep - constant step size
- ArmijoBacktracking - step dividing if the function value at the new point does not satisfy armijo condition.
- BisectionWolfe - bisection method that either computes a step size satisfying the weak Wolfe conditions or sends the function values to -inf.
