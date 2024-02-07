"""
Infinite-horizon job search with Markov wage draws.

"""

from quantecon.markov import tauchen
import numpy as np
from collections import namedtuple
from s_approx import successive_approx


# NamedTuple Model
Model = namedtuple("Model", ("n", "w_vals", "P", "β", "c"))

def create_markov_js_model(
        n=200,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.98,      # discount factor
        c=1.0        # unemployment compensation
    ):
    """
    Creates an instance of the job search model with Markov wages.
    """
    mc = tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P
    return Model(n=n, w_vals=w_vals, P=P, β=β, c=c)


def T(v, model):
    """
    The Bellman operator Tv = max{e, c + β P v} with e(w) = w / (1-β).
    """
    n, w_vals, P, β, c = model
    h = c + β * np.dot(P, v)
    e = w_vals / (1 - β)
    return np.maximum(e, h)


def get_greedy(v, model):
    """Get a v-greedy policy."""
    n, w_vals, P, β, c = model
    σ = w_vals / (1 - β) >= c + β * np.dot(P, v)
    return σ



def vfi(model):
    """Solve the infinite-horizon Markov job search model by VFI."""
    v_init = np.zeros(model.w_vals.shape)
    v_star = successive_approx(lambda v: T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star



# == Policy iteration == #


def get_value(σ, model):
    """Get the value of policy σ."""
    n, w_vals, P, β, c = model
    e = w_vals / (1 - β)
    K_σ = β * ((1 - σ) * P.T).T
    r_σ = σ * e + (1 - σ) * c
    I = np.identity(K_σ.shape[0])
    return np.linalg.solve((I - K_σ), r_σ)


def policy_iteration(model):
    """
    Howard policy iteration routine.
    """
    σ = np.zeros(model.n, dtype=bool)
    i, error = 0, True
    while error:
        v_σ = get_value(σ, model)
        σ_new = get_greedy(v_σ, model)
        error = np.any(σ_new ^ σ)
        σ = σ_new
        i = i + 1
        print(f"Concluded loop {i} with error: {error}.")
    return σ


# == Plots == #

import matplotlib.pyplot as plt


default_model = create_markov_js_model()


def plot_main(model=default_model,
               method="vfi",
               savefig=False,
               figname="./figures/markov_js_vfix.png"):
    n, w_vals, P, β, c = model

    if method == "vfi":
        v_star, σ_star = vfi(model)
    else:
        σ_star = policy_iteration(model)
        v_star = get_value(σ_star, model)

    h_star = c + β * np.dot(P, v_star)
    e = w_vals / (1 - β)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_vals, h_star, linewidth=4, ls="--", alpha=0.4, label=r"$h^*(w)$")
    ax.plot(w_vals, e, linewidth=4, ls="--", alpha=0.4, label=r"$w/(1-\beta)$")
    ax.plot(w_vals, np.maximum(e, h_star), "k-", alpha=0.7, label=r"$v^*(w)$")
    ax.legend(frameon=False)
    ax.set_xlabel(r"$w$")
    plt.show()
    if savefig:
        fig.savefig(figname)
