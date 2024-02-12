"""
Epstein--Zin utility: solving the recursion for a given consumption
path.

"""

from quantecon import compute_fixed_point
from quantecon.markov import tauchen

import numpy as np
from numba import njit
from collections import namedtuple


# NamedTuple Model
Model = namedtuple("Model", ("β", "ρ", "σ", "α", "γ", "c", "x_vals", "P"))


def create_ez_utility_model(
        n=200,      # size of state space
        ρ=0.96,     # correlation coef in AR(1)
        σ=0.1,      # volatility
        β=0.99,     # time discount factor
        α=0.75,     # EIS parameter
        γ=-2.0):    # risk aversion parameter
    mc = tauchen(n, ρ, σ, 0, 5)
    x_vals, P = mc.state_values, mc.P
    c = np.exp(x_vals)
    return Model(β=β, ρ=ρ, σ=σ, α=α, γ=γ, c=c, x_vals=x_vals, P=P)


@njit
def K(v, model):
    β, ρ, σ, α, γ, c, x_vals, P = model
    R = np.dot(P, v**γ)**(1/γ)
    return ((1 - β) * c**α + β * R**α)**(1/α)


def compute_ez_utility(model):
    v_init = np.ones(len(model.x_vals))
    v_star = compute_fixed_point(lambda v: K(v, model), v_init,
                                 error_tol=1e-6, max_iter=1000, print_skip=100)
    return v_star



# Plots


import matplotlib.pyplot as plt


def plot_convergence(savefig=False,
                     num_iter=100,
                     figname="./figures/ez_utility_c.pdf"):

    fig, ax = plt.subplots(figsize=(10, 5.2))
    model = create_ez_utility_model()
    β, ρ, σ, α, γ, c, x_vals, P = model


    v_star = compute_ez_utility(model)
    v = 0.1 * v_star
    ax.plot(x_vals, v, "k-", linewidth=3, alpha=0.7, label=r"$v_0$")

    greys = [str(g) for g in np.linspace(0.0, 0.4, num_iter)]
    greys.reverse()

    for g in greys:
        ax.plot(x_vals, v, "k-", linewidth=1, alpha=0.7)
        for t in range(20):
            v = K(v, model)

    v_star = compute_ez_utility(model)
    ax.plot(x_vals, v_star, linewidth=3, alpha=0.7, label=r"$v^*$")
    ax.set_xlabel(r"$x$")

    ax.legend(frameon=False, loc="upper left")
    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_v(savefig=False,
           figname="./figures/ez_utility_1.pdf"):

    fig, ax = plt.subplots(figsize=(10, 5.2))
    model = create_ez_utility_model()
    β, ρ, σ, α, γ, c, x_vals, P = model
    v_star = compute_ez_utility(model)
    ax.plot(x_vals, v_star, linewidth=2, alpha=0.7, label=r"$v^*$")
    ax.set_xlabel(r"$x$")

    ax.legend(frameon=False, loc="upper left")
    plt.show()
    if savefig:
        fig.savefig(figname)


def vary_gamma(gamma_vals=[1.0, -8.0],
               savefig=False,
               figname="./figures/ez_utility_2.pdf"):

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for γ in gamma_vals:
        model = create_ez_utility_model(γ=γ)
        β, ρ, σ, α, γ, c, x_vals, P = model
        v_star = compute_ez_utility(model)
        ax.plot(x_vals, v_star, linewidth=2, alpha=0.7, label=r"$\gamma=$" + f"{γ}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$v(x)$")

    ax.legend(frameon=False, loc="upper left")
    plt.show()
    if savefig:
        fig.savefig(figname)


def vary_alpha(alpha_vals=[0.5, 0.6],
               savefig=False,
               figname="./figures/ez_utility_3.pdf"):

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for α in alpha_vals:
        model = create_ez_utility_model(α=α)
        β, ρ, σ, α, γ, c, x_vals, P = model
        v_star = compute_ez_utility(model)
        ax.plot(x_vals, v_star, linewidth=2, alpha=0.7, label=r"$\alpha=$"+f"{α}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$v(x)$")

    ax.legend(frameon=False, loc="upper left")
    plt.show()
    if savefig:
        fig.savefig(figname)
