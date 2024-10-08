import numpy as np
import matplotlib.pyplot as plt
from quantecon.markov import tauchen
from numpy.linalg import solve


def compute_v(n=25, β=0.98, ρ=0.96, ν=0.05, γ=2.0):
    mc = tauchen(n, ρ, ν)
    x_vals = mc.state_values
    P = mc.P
    r = np.exp((1 - γ) * x_vals) / (1 - γ)
    v = solve(np.eye(n) - β * P, r)
    return x_vals, v


def plot_v(savefig=False, figname="../figures_py/val_consumption_1.png"):
    fontsize = 12

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x_vals, v = compute_v()
    ax.plot(x_vals, v, lw=2, alpha=0.7, label=r"$v$")
    ax.set_xlabel(r"$x$", fontsize=fontsize)
    ax.legend(frameon=False, fontsize=fontsize, loc="upper left")
    ax.set_ylim(-65, -40)
    plt.show()
    if savefig:
        fig.savefig(figname)


plot_v(savefig=True)
