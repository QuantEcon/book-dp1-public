import numpy as np

# Primitives
A = np.array([
    [0.4, 0.1],
    [0.7, 0.2]
])


# Method one: direct inverse
I = np.identity(2)
B_inverse = np.linalg.inv(I - A)


# Method two: power series
def power_series(A):
    B_sum = np.zeros((2, 2))
    A_power = np.identity(2)
    for k in range(50):
        B_sum += A_power
        A_power = np.dot(A_power, A)
    return B_sum


# Print maximal error
print(np.max(np.abs(B_inverse - power_series(A))))
