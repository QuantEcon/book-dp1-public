"""
VFI approach to job search in the infinite-horizon IID case.

"""

from quantecon import compute_fixed_point

from two_period_job_search import create_job_search_model

from numba import njit
import numpy as np


# A model with default parameters
default_model = create_job_search_model()


@njit
def T(v, model):
    """ The Bellman operator. """
    n, w_vals, φ, β, c = model
    return np.array([np.maximum(w / (1 - β),
                    c + β * np.sum(v * φ)) for w in w_vals])


@njit
def get_greedy(v, model):
    """ Get a v-greedy policy. """
    n, w_vals, φ, β, c = model
    σ = w_vals / (1 - β) >= c + β * np.sum(v * φ)  # Boolean policy vector
    return σ


def vfi(model=default_model):
    """ Solve the infinite-horizon IID job search model by VFI. """
    v_init = np.zeros_like(model.w_vals)
    v_star = compute_fixed_point(lambda v: T(v, model), v_init,
                                 error_tol=1e-5, max_iter=1000, print_skip=25)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star



# == Plots == #

import matplotlib.pyplot as plt


def fig_vseq(model=default_model,
                k=3,
                savefig=False,
                figname="./figures/iid_job_search_1.pdf",
                fs=10):

    v = np.zeros_like(model.w_vals)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i in range(k):
        ax.plot(model.w_vals, v, linewidth=3, alpha=0.6,
                label=f"iterate {i}")
        v = T(v, model)

    for i in range(1000):
        v = T(v, model)

    ax.plot(model.w_vals, v, "k-", linewidth=3.0,
            label="iterate 1000", alpha=0.7)

    fontdict = {'fontsize': fs}
    ax.set_xlabel("wage offer", fontdict=fontdict)
    ax.set_ylabel("lifetime value", fontdict=fontdict)

    ax.legend(fontsize=fs, frameon=False)

    if savefig:
        fig.savefig(figname)
    plt.show()


def fig_vstar(model=default_model,
              savefig=False, fs=10,
              figname="./figures/iid_job_search_3.pdf"):
    """ Plot the fixed point. """
    n, w_vals, φ, β, c = model
    v_star, σ_star = vfi(model)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(w_vals, v_star, "k-", linewidth=1.5, label="value function")
    cont_val = c + β * np.sum(v_star * φ)
    ax.plot(w_vals, [cont_val]*(n+1),
            "--",
            linewidth=5,
            alpha=0.5,
            label="continuation value")

    ax.plot(w_vals,
            w_vals / (1 - β),
            "--",
            linewidth=5,
            alpha=0.5,
            label=r"$w/(1 - \beta)$")

    ax.legend(frameon=False, fontsize=fs, loc="lower right")

    if savefig:
        fig.savefig(figname)
    plt.show()
