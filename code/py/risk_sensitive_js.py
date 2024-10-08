"""
Infinite-horizon job search with Markov wage draws and risk-sensitive preferences.

"""

from quantecon import compute_fixed_point
from quantecon.markov import tauchen

import numpy as np
from numba import njit
from collections import namedtuple
import matplotlib.pyplot as plt

# NamedTuple Model
Model = namedtuple("Model", ("n", "w_vals", "P", "β", "c", "θ"))


def create_markov_js_model(
        n=200,        # wage grid size
        ρ=0.9,        # wage persistence
        ν=0.2,        # wage volatility
        β=0.98,       # discount factor
        c=1.0,        # unemployment compensation
        θ=-0.01       # risk parameter
    ):
    """Creates an instance of the job search model with Markov wages."""
    mc = tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P
    return Model(n=n, w_vals=w_vals, P=P, β=β, c=c, θ=θ)


@njit
def T(v, model):
    """
    The Bellman operator Tv = max{e, c + β R v} with

    e(w) = w / (1-β) and

    (Rv)(w) = (1/θ) ln{E_w[ exp(θ v(W'))]}

    """
    n, w_vals, P, β, c, θ = model
    h = c + (β / θ) * np.log(np.dot(P, (np.exp(θ * v))))
    e = w_vals / (1 - β)
    return np.maximum(e, h)


@njit
def get_greedy(v, model):
    """Get a v-greedy policy."""
    n, w_vals, P, β, c, θ = model
    σ = w_vals / (1 - β) >= c + (β / θ) * np.log(np.dot(P, (np.exp(θ * v))))
    return σ


def vfi(model):
    """Solve the infinite-horizon Markov job search model by VFI."""
    v_init = np.zeros(model.w_vals.shape)
    v_star = compute_fixed_point(lambda v: T(v, model), v_init,
                                 error_tol=1e-5, max_iter=1000, print_skip=25)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star


# == Plots == #

def plot_main(theta_vals=(-10, 0.0001, 0.1),
              savefig=False,
              figname="../figures_py/risk_sensitive_js.png"):

    fig, axes = plt.subplots(len(theta_vals), 1, figsize=(9, 22))

    for (θ, ax) in zip(theta_vals, axes):
        model = create_markov_js_model(θ=θ)
        n, w_vals, P, β, c, θ = model
        v_star, σ_star = vfi(model)

        h_star = c + (β / θ) * np.log(np.dot(P, (np.exp(θ * v_star))))
        e = w_vals / (1 - β)

        ax.plot(w_vals, h_star, linewidth=4, ls="--", alpha=0.4, label=r"$h^*(w)$")
        ax.plot(w_vals, e, linewidth=4, ls="--", alpha=0.4, label=r"$w/(1-\beta)$")
        ax.plot(w_vals, np.maximum(e, h_star), "k-", alpha=0.7, label=r"$v^*(w)$")
        ax.set_title(r"$\theta = $" + f"{θ}")
        ax.legend(frameon=False)
        ax.set_xlabel(r"$w$")

    fig.tight_layout()
    plt.show()

    if savefig:
        fig.savefig(figname)

plot_main(savefig=True)
