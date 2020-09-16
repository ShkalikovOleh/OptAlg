# OptAlg
Set of optimization algorithms.

## Iterative
Methods based on iteration over values and calculating the function for each of them.

- [SimpleSearch](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/iterative/simple_search.py) - uniform selection of n values from the interval.

## Gradient
Methods based on descent to minimum by gradient-like direction.

- [Gradient descent with step decrease](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/gradient/gd_step_decrease.py) - gradient descent with step dividing if the function value at the new point is greater than the function value at the previous point.

- [Gradient descent fastest](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/gradient/gd_fastest.py) - gradient descent with step determine by other optimizer.

- [Cojugate directions descent](https://github.com/ShkalikovOleh/OptAlg/blob/master/optalg/gradient/gd_conjugate.py) - descent direction is the sum of gradient in current point and the weighted direction from the previous iteration.