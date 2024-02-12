import numpy as np
from scipy.stats import geom
from itertools import product
from quantecon import MarkovChain
from collections import namedtuple

# NamedTuple Model
Model = namedtuple("Model", ("S", "s", "p", "φ", "h"))

def create_inventory_model(S=100,   # Order size
                           s=10,    # Order threshold
                           p=0.4):  # Demand parameter
    φ = geom(p, loc=-1) # loc sets support to {0,1,...}
    h = lambda x, d: max(x - d, 0) + S*(x <= s)
    return Model(S=S, s=s, p=p, φ=φ, h=h)


def sim_inventories(model, ts_length=200):
    """Simulate the inventory process."""
    S, s, p, φ, h = model
    X = np.empty(ts_length)
    X[0] = S  # Initial condition
    for t in range(0, ts_length - 1):
        X[t+1] = h(X[t], φ.rvs())
    return X


def compute_mc(model, d_max=100):
    """Compute the transition probabilities and state."""
    S, s, p, φ, h = model
    n = S + s + 1  # Size of state space
    state_vals = np.arange(n)
    P = np.empty((n, n))
    for (i, j) in product(range(0, n), range(0, n)):
        P[i, j] = sum((h(i, d) == j)*φ.pmf(d) for d in range(d_max+1))
    return MarkovChain(P, state_vals)


def compute_stationary_dist(model):
    """Compute the stationary distribution of the model."""
    mc = compute_mc(model)
    return mc.state_values, mc.stationary_distributions[0]



# Plots

import matplotlib.pyplot as plt


default_model = create_inventory_model()


def plot_ts(model, fontsize=16,
                   figname="./figures/inventory_sim_1.pdf",
                   savefig=False):
    S, s, p, φ, h = model
    X = sim_inventories(model)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(X, label=r"$X_t$", linewidth=3, alpha=0.6)
    fontdict = {'fontsize': fontsize}
    ax.set_xlabel(r"$t$", fontdict=fontdict)
    ax.set_ylabel("inventory", fontdict=fontdict)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_ylim(0, S + s + 20)

    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_hist(model, fontsize=16,
                   figname="./figures/inventory_sim_2.pdf",
                   savefig=False):
    S, s, p, φ, h = model
    state_values, ψ_star = compute_stationary_dist(model)
    X = sim_inventories(model, 1_000_000)
    histogram = [np.mean(X == i) for i in state_values]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(state_values, ψ_star, "k-",  linewidth=3, alpha=0.7,
                label=r"$\psi^*$")
    ax.bar(state_values, histogram, alpha=0.7, label="frequency")
    fontdict = {'fontsize': fontsize}
    ax.set_xlabel("state", fontdict=fontdict)

    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_ylim(0, 0.015)

    plt.show()
    if savefig:
        fig.savefig(figname)
