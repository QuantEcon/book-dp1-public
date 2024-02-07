import numpy as np
from quantecon.markov import tauchen, MarkovChain

from collections import namedtuple
from numba import njit, prange


# NamedTuple Model
Model = namedtuple("Model", ("β", "κ", "α", "p", "w", "l_grid",
                             "z_grid", "Q"))


def create_hiring_model(
        r=0.04,                              # Interest rate
        κ=1.0,                               # Adjustment cost
        α=0.4,                               # Production parameter
        p=1.0, w=1.0,                        # Price and wage
        l_min=0.0, l_max=30.0, l_size=100,   # Grid for labor
        ρ=0.9, ν=0.4, b=1.0,                 # AR(1) parameters
        z_size=100):                         # Grid size for shock
    β = 1/(1+r)
    l_grid = np.linspace(l_min, l_max, l_size)
    mc = tauchen(z_size, ρ, ν, b, 6)
    z_grid, Q = mc.state_values, mc.P
    return Model(β=β, κ=κ, α=α, p=p, w=w,
                 l_grid=l_grid, z_grid=z_grid, Q=Q)


@njit
def B(i, j, k, v, model):
    """
    The aggregator B is given by

        B(l, z, l′) = r(l, z, l′) + β Σ_z′ v(l′, z′) Q(z, z′)."

    where

        r(l, z, l′) := p * z * f(l) - w * l - κ 1{l != l′}

    """
    β, κ, α, p, w, l_grid, z_grid, Q = model
    l, z, l_1 = l_grid[i], z_grid[j], l_grid[k]
    r = p * z * l**α - w * l - κ * (l != l_1)
    return r + β * np.dot(v[k, :], Q[j, :])


@njit(parallel=True)
def T_σ(v, σ, model):
    """The policy operator."""
    v_new = np.empty_like(v)
    for i in prange(len(model.l_grid)):
        for j in prange(len(model.z_grid)):
            v_new[i, j] = B(i, j, σ[i, j], v, model)
    return v_new


@njit(parallel=True)
def get_greedy(v, model):
    """Compute a v-greedy policy."""
    β, κ, α, p, w, l_grid, z_grid, Q = model
    n, m = len(l_grid), len(z_grid)
    σ = np.empty((n, m), dtype=np.int32)
    for i in prange(n):
        for j in prange(m):
            tmp = np.array([B(i, j, k, v, model) for k
                            in np.arange(n)])
            σ[i, j] = np.argmax(tmp)
    return σ


@njit
def optimistic_policy_iteration(model, tolerance=1e-5, m=100):
    """Optimistic policy iteration routine."""
    v = np.zeros((len(model.l_grid), len(model.z_grid)))
    error = tolerance + 1
    while error > tolerance:
        last_v = v
        σ = get_greedy(v, model)
        for i in range(m):
            v = T_σ(v, σ, model)
        error = np.max(np.abs(v - last_v))
    return get_greedy(v, model)


# Plots

import matplotlib.pyplot as plt


def plot_policy(savefig=False,
                figname="./figures/firm_hiring_pol.pdf"):
    model = create_hiring_model()
    β, κ, α, p, w, l_grid, z_grid, Q = model
    σ_star = optimistic_policy_iteration(model)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(l_grid, l_grid, "k--", label=r"$45$")
    ax.plot(l_grid, l_grid[σ_star[:, 0]], label=r"$\sigma^*(\cdot, z_1)$")
    ax.plot(l_grid, l_grid[σ_star[:, -1]], label=r"$\sigma^*(\cdot, z_N)$")
    ax.legend()
    plt.show()
    if savefig:
        fig.savefig(figname)


def sim_dynamics(model, ts_length):
    β, κ, α, p, w, l_grid, z_grid, Q = model
    σ_star = optimistic_policy_iteration(model)
    mc = MarkovChain(Q, z_grid)
    z_sim_idx = mc.simulate_indices(ts_length)
    z_sim = z_grid[z_sim_idx]
    l_sim_idx = np.empty(ts_length, dtype=np.int32)
    l_sim_idx[0] = 32
    for t in range(ts_length-1):
        l_sim_idx[t+1] = σ_star[l_sim_idx[t], z_sim_idx[t]]
    l_sim = l_grid[l_sim_idx]

    y_sim = np.empty_like(l_sim)
    for (i, l) in enumerate(l_sim):
        y_sim[i] = p * z_sim[i] * l_sim[i]**α

    t = ts_length - 1
    l_g, y_g, z_g = np.zeros(t), np.zeros(t), np.zeros(t)

    for i in range(t):
        l_g[i] = (l_sim[i+1] - l_sim[i]) / l_sim[i]
        y_g[i] = (y_sim[i+1] - y_sim[i]) / y_sim[i]
        z_g[i] = (z_sim[i+1] - z_sim[i]) / z_sim[i]

    return l_sim, y_sim, z_sim, l_g, y_g, z_g


def plot_sim(savefig=False,
             figname="./figures/firm_hiring_ts.pdf",
             ts_length = 250):
    model = create_hiring_model()
    β, κ, α, p, w, l_grid, z_grid, Q = model
    l_sim, y_sim, z_sim, l_g, y_g, z_g = sim_dynamics(model, ts_length)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x_grid = np.arange(ts_length)
    ax.plot(x_grid, l_sim, label=r"$\ell_t$")
    ax.plot(x_grid, z_sim, alpha=0.6, label=r"$Z_t$")
    ax.legend(frameon=False)
    ax.set_ylabel("employment")
    ax.set_xlabel("time")

    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_growth(savefig=False,
                figname="./figures/firm_hiring_g.pdf",
                ts_length = 10_000_000):

    model = create_hiring_model()
    β, κ, α, p, w, l_grid, z_grid, Q = model
    l_sim, y_sim, z_sim, l_g, y_g, z_g = sim_dynamics(model, ts_length)

    fig, ax = plt.subplots()
    ax.hist(l_g, alpha=0.6, bins=100)
    ax.set_xlabel("growth")

    #fig, axes = plt.subplots(2, 1)
    #series = y_g, z_g
    #for (ax, g) in zip(axes, series):
    #    ax.hist(g, alpha=0.6, bins=100)
    #    ax.set_xlabel("growth")

    plt.tight_layout()
    plt.show()
    if savefig:
        fig.savefig(figname)
