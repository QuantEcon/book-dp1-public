import numpy as np
import matplotlib.pyplot as plt

def plot_policy(σ, model, title="", savefig=False, figname="../figures_py/ez_policies.png", fontsize=16):
    w_grid = model.w_grid
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, w_grid, "k--", label=r"$45$")
    ax.plot(w_grid, w_grid[σ[:, 0]], label=r"$\sigma^*(\cdot, e_1)$")
    ax.plot(w_grid, w_grid[σ[:, -1]], label=r"$\sigma^*(\cdot, e_N)$")
    ax.legend(fontsize=fontsize)
    ax.set_xlabel("$w$", fontsize=fontsize)
    ax.set_ylabel("$\sigma^*$", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    if savefig:
        plt.savefig(figname)
    plt.show()

def plot_value_orig(v, model, fontsize=16):
    w_grid = model.w_grid
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, v[:, 0], label=r"$v^*(\cdot, e_1)$")
    ax.plot(w_grid, v[:, -1], label=r"$v^*(\cdot, e_N)$")
    ax.legend(fontsize=fontsize)
    ax.set_xlabel("$w$", fontsize=fontsize)
    ax.set_ylabel("$v^*$", fontsize=fontsize)
    plt.show()

def plot_value_mod(h, model, fontsize=16):
    w_grid = model.w_grid
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, h, label=r"$h^*$")
    ax.legend(fontsize=fontsize)
    ax.set_xlabel("$w$", fontsize=fontsize)
    ax.set_ylabel("$h^*$", fontsize=fontsize)
    plt.show()
