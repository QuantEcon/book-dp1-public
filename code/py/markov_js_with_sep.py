"""
Infinite-horizon job search with Markov wage draws and separation.

"""

from quantecon.markov import tauchen
import numpy as np
from collections import namedtuple
from s_approx import successive_approx


# NamedTuple Model
Model = namedtuple("Model", ("n", "w_vals", "P", "β", "c", "α"))


def create_js_with_sep_model(
        n=200,          # wage grid size
        ρ=0.9, ν=0.2,   # wage persistence and volatility
        β=0.98, α=0.1,  # discount factor and separation rate
        c=1.0):         # unemployment compensation
    """Creates an instance of the job search model with separation."""
    mc = tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P
    return Model(n=n, w_vals=w_vals, P=P, β=β, c=c, α=α)


def T(v, model):
    """The Bellman operator for the value of being unemployed."""
    n, w_vals, P, β, c, α = model
    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * np.dot(P, v))
    reject = c + β * np.dot(P, v)
    return np.maximum(accept, reject)


def get_greedy(v, model):
    """ Get a v-greedy policy."""
    n, w_vals, P, β, c, α = model
    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * np.dot(P, v))
    reject = c + β * np.dot(P, v)
    σ = accept >= reject
    return σ


def vfi(model):
    """Solve by VFI."""
    v_init = np.zeros(model.w_vals.shape)
    v_star = successive_approx(lambda v: T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star



# == Plots == #

import matplotlib.pyplot as plt


default_model = create_js_with_sep_model()


def plot_main(model=default_model,
              savefig=False,
              figname="./figures/markov_js_with_sep_1.pdf"):
    n, w_vals, P, β, c, α = model
    v_star, σ_star = vfi(model)

    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * np.dot(P, v_star))
    h_star = c + β * np.dot(P, v_star)

    w_star = np.inf
    for (i, w) in enumerate(w_vals):
        if accept[i] >= h_star[i]:
            w_star = w
            break

    assert w_star != np.inf, "Agent never accepts"

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_vals, h_star, linewidth=4, ls="--", alpha=0.4,
            label="continuation value")
    ax.plot(w_vals, accept, linewidth=4, ls="--", alpha=0.4,
            label="stopping value")
    ax.plot(w_vals, v_star, "k-", alpha=0.7, label=r"$v_u^*(w)$")
    ax.legend(frameon=False)
    ax.set_xlabel(r"$w$")
    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_w_stars(α_vals=np.linspace(0.0, 1.0, 10),
                 savefig=False,
                 figname="./figures/markov_js_with_sep_2.pdf"):

    w_star_vec = np.empty_like(α_vals)
    for (i_α, α) in enumerate(α_vals):
        print(i_α, α)
        model = create_js_with_sep_model(α=α)
        n, w_vals, P, β, c, α = model
        v_star, σ_star = vfi(model)

        d = 1 / (1 - β * (1 - α))
        accept = d * (w_vals + α * β * np.dot(P, v_star))
        h_star = c + β * np.dot(P, v_star)

        w_star = np.inf
        for (i_w, w) in enumerate(w_vals):
            if accept[i_w] >= h_star[i_w]:
                w_star = w
                break

        assert w_star != np.inf, "Agent never accepts"
        w_star_vec[i_α] = w_star

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(α_vals, w_star_vec, linewidth=2, alpha=0.6,
            label="reservation wage")
    ax.legend(frameon=False)
    ax.set_xlabel(r"$\alpha$")
    ax.set_xlabel(r"$w$")
    plt.show()
    if savefig:
        fig.savefig(figname)
