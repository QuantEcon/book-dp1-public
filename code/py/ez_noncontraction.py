import numpy as np
import matplotlib.pyplot as plt

def F(w, r=0.5, β=0.5, θ=5):
    return (r + β * w**(1/θ))**θ

w_grid = np.linspace(0.001, 2.0, 200)

def plot_F(savefig=False, figname="../figures_py/ez_noncontraction.png", fs=16):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    f = lambda w: F(w, θ=-10)
    ax.plot(w_grid, w_grid, "k--", alpha=0.6, label=r"$45$")
    ax.plot(w_grid, f(w_grid), label=r"$\hat{K} = F$")
    ax.set_xticks((0, 1, 2))
    ax.set_yticks((0, 1, 2))
    ax.legend(fontsize=fs, frameon=False)

    plt.show()

    if savefig:
        fig.savefig(figname)

plot_F()
