import numpy as np

# Spectral radius
ρ = lambda A: np.max(np.abs(np.linalg.eigvals(A)))

# Test with arbitrary A
A = np.array([
    [0.4, 0.1],
    [0.7, 0.2]
])
print(ρ(A))
