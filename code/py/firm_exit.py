"""
Firm valuation with exit option.

"""

from quantecon.markov import tauchen
from quantecon import compute_fixed_point

import numpy as np
from collections import namedtuple
from numba import njit


# NamedTuple Model
Model = namedtuple("Model", ("n", "z_vals", "Q", "β", "s"))


def create_exit_model(
        n=200,                  # productivity grid size
        ρ=0.95, μ=0.1, ν=0.1,   # persistence, mean and volatility
        β=0.98, s=100.0         # discount factor and scrap value
    ):
    """
    Creates an instance of the firm exit model.
    """
    mc = tauchen(n, ρ, ν, mu=μ)
    z_vals, Q = mc.state_values, mc.P
    return Model(n=n, z_vals=z_vals, Q=Q, β=β, s=s)


@njit
def no_exit_value(model):
    """Compute value of firm without exit option."""
    n, z_vals, Q, β, s = model
    I = np.identity(n)
    return np.linalg.solve((I - β * Q), z_vals)


@njit
def T(v, model):
    """The Bellman operator Tv = max{s, π + β Q v}."""
    n, z_vals, Q, β, s = model
    h = z_vals + β * np.dot(Q, v)
    return np.maximum(s, h)


@njit
def get_greedy(v, model):
    """Get a v-greedy policy."""
    n, z_vals, Q, β, s = model
    σ = s >= z_vals + β * np.dot(Q, v)
    return σ


def vfi(model):
    """Solve by VFI."""
    v_init = no_exit_value(model)
    v_star = compute_fixed_point(lambda v: T(v, model), v_init, error_tol=1e-6,
                                 max_iter=1000, print_skip=25)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star


# Plots


import matplotlib.pyplot as plt


def plot_val(savefig=False,
             figname="../figures_py/firm_exit_1.png"):

    fig, ax = plt.subplots(figsize=(9, 5.2))

    model = create_exit_model()
    n, z_vals, Q, β, s = model

    v_star, σ_star = vfi(model)
    h = z_vals + β * np.dot(Q, v_star)

    ax.plot(z_vals, h, "-", linewidth=3, alpha=0.6, label=r"$h^*$")
    ax.plot(z_vals, s * np.ones(n), "-", linewidth=3, alpha=0.6, label=r"$s$")
    ax.plot(z_vals, v_star, "k--", linewidth=1.5, alpha=0.8, label=r"$v^*$")

    ax.legend(frameon=False)
    ax.set_xlabel(r"$z$")

    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_comparison(savefig=False,
                    figname="../figures_py/firm_exit_2.png"):

    fig, ax = plt.subplots(figsize=(9, 5.2))

    model = create_exit_model()
    n, z_vals, Q, β, s = model

    v_star, σ_star = vfi(model)
    w = no_exit_value(model)

    ax.plot(z_vals, v_star, "k-", linewidth=2, alpha=0.6, label="$v^*$")
    ax.plot(z_vals, w, linewidth=2, alpha=0.6, label="no-exit value")

    ax.legend(frameon=False)
    ax.set_xlabel(r"$z$")

    plt.show()
    if savefig:
        fig.savefig(figname)

plot_val(savefig=True)
plot_comparison(savefig=True)
