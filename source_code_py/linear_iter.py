from s_approx import successive_approx
import numpy as np

# Compute the fixed point of Tx = Ax + b via linear algebra
A = np.array([
    [0.4, 0.1],
    [0.7, 0.2]
])

b = np.array([
    [1.0],
    [2.0]
])

I = np.identity(2)
x_star = np.linalg.solve(I - A, b)  # compute (I - A)^{-1} * b


# Compute the fixed point via successive approximation
T = lambda x: np.dot(A, x) + b
x_0 = np.array([
    [1.0],
    [1.0]
])
x_star_approx = successive_approx(T, x_0)

# Test for approximate equality (prints "True")
print(np.allclose(x_star, x_star_approx, rtol=1e-5))
