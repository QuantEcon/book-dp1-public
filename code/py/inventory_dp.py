from quantecon import compute_fixed_point

import numpy as np
from collections import namedtuple
from numba import njit


# NamedTuple Model
Model = namedtuple("Model", ("β", "K", "c", "κ", "p"))


def create_inventory_model(β=0.98,      # discount factor
                           K=40,        # maximum inventory
                           c=0.2, κ=2,  # cost parameters
                           p=0.6):      # demand parameter
    return Model(β=β, K=K, c=c, κ=κ, p=p)


@njit
def demand_pdf(d, p):
    return (1 - p)**d * p


@njit
def B(x, a, v, model, d_max=101):
    """
    The function B(x, a, v) = r(x, a) + β Σ_x′ v(x′) P(x, a, x′).
    """
    β, K, c, κ, p = model
    x1 = np.array([np.minimum(x, d)*demand_pdf(d, p) for d in np.arange(d_max)])
    reward = np.sum(x1) - c * a - κ * (a > 0)
    x2 = np.array([v[np.maximum(0, x - d) + a] * demand_pdf(d, p)
                                 for d in np.arange(d_max)])
    continuation_value = β * np.sum(x2)
    return reward + continuation_value


@njit
def T(v, model):
    """The Bellman operator."""
    β, K, c, κ, p = model
    new_v = np.empty_like(v)
    for x in range(0, K+1):
        x1 = np.array([B(x, a, v, model) for a in np.arange(K-x+1)])
        new_v[x] = np.max(x1)
    return new_v


@njit
def get_greedy(v, model):
    """
    Get a v-greedy policy.  Returns a zero-based array.
    """
    β, K, c, κ, p = model
    σ_star = np.zeros(K+1, dtype=np.int32)
    for x in range(0, K+1):
        x1 = np.array([B(x, a, v, model) for a in np.arange(K-x+1)])
        σ_star[x] = np.argmax(x1)
    return σ_star


def solve_inventory_model(v_init, model):
    """Use successive_approx to get v_star and then compute greedy."""
    β, K, c, κ, p = model
    v_star = compute_fixed_point(lambda v: T(v, model), v_init,
                                 error_tol=1e-5, max_iter=1000, print_skip=25)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star


# == Plots == #

import matplotlib.pyplot as plt


# Create an instance of the model and solve it
model = create_inventory_model()
β, K, c, κ, p = model
v_init = np.zeros(K+1)
v_star, σ_star = solve_inventory_model(v_init, model)


def sim_inventories(ts_length=400, X_init=0):
    """Simulate given the optimal policy."""
    global p, σ_star
    X = np.zeros(ts_length, dtype=np.int32)
    X[0] = X_init
    # Subtracts 1 because numpy generates only positive integers
    rand = np.random.default_rng().geometric(p=p, size=ts_length-1) - 1
    for t in range(0, ts_length-1):
        X[t+1] = np.maximum(X[t] - rand[t], 0) + σ_star[X[t] + 1]
    return X


def plot_vstar_and_opt_policy(fontsize=10,
                   figname="./figures/inventory_dp_vs.pdf",
                   savefig=False):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))

    ax = axes[0]
    ax.plot(np.arange(K+1), v_star, label=r"$v^*$")
    ax.set_ylabel("value", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)

    ax = axes[1]
    ax.plot(np.arange(K+1), σ_star, label=r"$\sigma^*$")
    ax.set_xlabel("inventory", fontsize=fontsize)
    ax.set_ylabel("optimal choice", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)
    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_ts(fontsize=10,
            figname="./figures/inventory_dp_ts.pdf",
            savefig=False):
    X = sim_inventories()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(X, label="$X_t$", alpha=0.7)
    ax.set_xlabel("$t$", fontsize=fontsize)
    ax.set_ylabel("inventory", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_ylim(0, np.max(X)+4)
    plt.show()
    if savefig:
        fig.savefig(figname)
