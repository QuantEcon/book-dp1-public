from quantecon import compute_fixed_point
from quantecon.markov import tauchen

import numpy as np
from numba import njit
from collections import namedtuple


# NamedTuple Model
Model = namedtuple("Model", ("β", "θ", "ρ", "σ", "r", "x_vals", "P"))


def create_rs_utility_model(
        n=180,      # size of state space
        β=0.95,     # time discount factor
        ρ=0.96,     # correlation coef in AR(1)
        σ=0.1,      # volatility
        θ=-1.0):    # risk aversion
    mc = tauchen(n, ρ, σ, 0, 10)  # n_std = 10
    x_vals, P = mc.state_values, mc.P
    r = x_vals      # special case u(c(x)) = x
    return Model(β=β, θ=θ, ρ=ρ, σ=σ, r=r, x_vals=x_vals, P=P)


@njit
def K(v, model):
    β, θ, ρ, σ, r, x_vals, P = model
    return r + (β/θ) * np.log(np.dot(P, (np.exp(θ*v))))


def compute_rs_utility(model):
    β, θ, ρ, σ, r, x_vals, P = model
    v_init = np.zeros(len(x_vals))
    v_star = compute_fixed_point(lambda v: K(v, model), v_init,
                                 error_tol=1e-10, max_iter=1000, print_skip=25)
    return v_star


# Plots


import matplotlib.pyplot as plt


def plot_v(savefig=False,
           figname="./figures/rs_utility_1.pdf"):

    fig, ax = plt.subplots(figsize=(10, 5.2))
    model = create_rs_utility_model()
    β, θ, ρ, σ, r, x_vals, P = model

    a = 1/(1 - (ρ*β))
    b = (β /(1 - β)) * (θ/2) * (a*σ)**2

    v_star = compute_rs_utility(model)
    v_star_a = a * x_vals + b
    ax.plot(x_vals, v_star, linewidth=2, alpha=0.7,
            label="approximate fixed point")
    ax.plot(x_vals, v_star_a, "k--", linewidth=2, alpha=0.7,
            label=r"$v(x)=ax + b$")
    ax.set_xlabel(r"$x$")

    ax.legend(frameon=False, loc="upper left")
    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_multiple_v(savefig=False,
                    figname="./figures/rs_utility_2.pdf"):

    fig, ax = plt.subplots(figsize=(10, 5.2))
    σ_vals = 0.05, 0.1

    for σ in σ_vals:
        model = create_rs_utility_model(σ=σ)
        β, θ, ρ, σ, r, x_vals, P  = model
        v_star = compute_rs_utility(model)
        ax.plot(x_vals, v_star, linewidth=2, alpha=0.7,
                label=r"$\sigma=$" + f"{σ}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$v(x)$")

    ax.legend(frameon=False, loc="upper left")
    plt.show()
    if savefig:
        fig.savefig(figname)
