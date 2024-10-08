import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from quantecon import tauchen

fontsize = 18

# Spectral radius
@njit
def ρ(A):
    return np.max(np.abs(np.linalg.eigvals(A)))

def plot_contours(savefig=False, figname="../figures_py/discount_spec_rad.png"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    grid_size = 50
    n = 6
    a_vals = np.linspace(0.1, 0.95, grid_size)
    s_vals = np.linspace(0.1, 0.32, grid_size)
    μ = 0.96

    R = np.zeros((grid_size, grid_size))
    L = np.zeros((n, n))

    for i_a, a in enumerate(a_vals):
        for i_s, s in enumerate(s_vals):
            mc = tauchen(n, a, np.sqrt(1 - a**2) * s, (1 - a) * μ)
            z_vals, Q = mc.state_values, mc.P
            β_vals = z_vals # np.maximum(z_vals, 0)
            L = β_vals * Q
            R[i_a, i_s] = ρ(L)

    cs1 = ax.contourf(a_vals, s_vals, R.T, alpha=0.5)
    ctr1 = ax.contour(a_vals, s_vals, R.T, levels=[1.0])
    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax)

    ax.set_xlabel(r"$a$", fontsize=fontsize)
    ax.set_ylabel(r"$s$", fontsize=fontsize)

    if savefig:
        fig.savefig(figname)
    plt.show()


plot_contours(True)
