"""
Price-dividend ratio in a model with dividend and consumption growth.

"""

from quantecon.markov import tauchen
import numpy as np
from collections import namedtuple


# NamedTuple Model
Model = namedtuple("Model", ("x_vals", "P", "β", "γ",
                            "μ_c", "σ_c", "μ_d", "σ_d"))


def create_asset_pricing_model(
        n=200,               # state grid size
        ρ=0.9, ν=0.2,        # state persistence and volatility
        β=0.99, γ=2.5,       # discount and preference parameter
        μ_c=0.01, σ_c=0.02,  # consumption growth mean and volatility
        μ_d=0.02, σ_d=0.1):  # dividend growth mean and volatility
    """
    Creates an instance of the asset pricing model with Markov state.
    """
    mc = tauchen(n, ρ, ν)
    x_vals, P = np.exp(mc.state_values), mc.P
    return Model(x_vals=x_vals, P=P, β=β, γ=γ,
                 μ_c=μ_c, σ_c=σ_c, μ_d=μ_d, σ_d=σ_d)


def build_discount_matrix(model):
    """Build the discount matrix A."""
    x_vals, P, β, γ, μ_c, σ_c, μ_d, σ_d = model
    e = np.exp(μ_d - γ*μ_c + (γ**2 * σ_c**2 + σ_d**2)/2 + (1-γ)*x_vals)
    return β * (e * P.T).T



def pd_ratio(model):
    """
    Compute the price-dividend ratio associated with the model.
    """
    x_vals, P, β, γ, μ_c, σ_c, μ_d, σ_d = model
    A = build_discount_matrix(model)
    assert np.max(np.abs(np.linalg.eigvals(A))) < 1, "Requires r(A) < 1."
    n = len(x_vals)
    I = np.identity(n)
    return np.linalg.solve((I - A), np.dot(A, np.ones(n)))


# == Plots == #


import matplotlib.pyplot as plt


default_model = create_asset_pricing_model()


def plot_main(μ_d_vals=(0.02, 0.08),
              savefig=False,
              figname="./figures/pd_ratio_1.pdf"):
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for μ_d in μ_d_vals:
        model = create_asset_pricing_model(μ_d=μ_d)
        x_vals, P, β, γ, μ_c, σ_c, μ_d, σ_d = model
        v_star = pd_ratio(model)
        ax.plot(x_vals, v_star, linewidth=2, alpha=0.6,
                label=r"$\mu_d$=" + f"{μ_d}")

    ax.legend(frameon=False)
    ax.set_xlabel(r"$x$")
    plt.show()
    if savefig:
        fig.savefig(figname)
