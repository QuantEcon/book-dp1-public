import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def F(w, r=1, β=0.5, θ=5):
    return (r + β * w**(1/θ))**θ

w_grid = np.linspace(0.1, 2.0, 200)

fig, axes = plt.subplots(2, 2)

θ_vals = [-2, -0.5, 0.5, 2]

for i in range(2):
    for j in range(2):
        θ = θ_vals[i*2 + j]
        f = lambda w: F(w, θ=θ)
        axes[i, j].plot(w_grid, w_grid, "k--", alpha=0.6, label=r"$45$")
        axes[i, j].plot(w_grid, f(w_grid), label=r"$U$")
        axes[i, j].legend()
        axes[i, j].set_title(f"$\\theta = {θ}$")

fig.tight_layout()
plt.show()
