from quantecon import tauchen, MarkovChain

import numpy as np
from collections import namedtuple
from numba import njit, prange
from math import floor
import matplotlib.pyplot as plt


# NamedTuple Model
Model = namedtuple("Model", ("β", "γ", "η_grid", "φ",
                             "w_grid", "y_grid", "Q"))


def create_savings_model(β=0.98, γ=2.5,  
                         w_min=0.01, w_max=20.0, w_size=100,
                         ρ=0.9, ν=0.1, y_size=20,
                         η_min=0.75, η_max=1.25, η_size=2):
    η_grid = np.linspace(η_min, η_max, η_size)
    φ = np.ones(η_size) * (1 / η_size)  # Uniform distributoin
    w_grid = np.linspace(w_min, w_max, w_size)
    mc = tauchen(y_size, ρ, ν)
    y_grid, Q = np.exp(mc.state_values), mc.P
    return Model(β=β, γ=γ, η_grid=η_grid, φ=φ, w_grid=w_grid,
                 y_grid=y_grid, Q=Q)

## == Functions for regular OPI == ##

@njit
def U(c, γ):
    return c**(1-γ)/(1-γ)

@njit
def B(i, j, k, l, v, model):
    """
    The function

    B(w, y, η, w′) = u(w + y - w′/η)) + β Σ v(w′, y′, η′) Q(y, y′) ϕ(η′)

    """
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    w, y, η, w_1 = w_grid[i], y_grid[j], η_grid[k], w_grid[l]
    c = w + y - (w_1 / η)
    exp_value = 0.0
    for m in prange(len(y_grid)):
        for n in prange(len(η_grid)):
            exp_value += v[l, m, n] * Q[j, m] * φ[n]
    return U(c, γ) + β * exp_value if c > 0 else -np.inf


@njit(parallel=True)
def T_σ(v, σ, model):
    """The policy operator."""
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    v_new = np.empty_like(v)
    for i in prange(len(w_grid)):
        for j in prange(len(y_grid)):
            for k in prange(len(η_grid)):
                v_new[i, j, k] = B(i, j, k, σ[i, j, k], v, model)
    return v_new


@njit(parallel=True)
def get_greedy(v, model):
    """Compute a v-greedy policy."""
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    w_n, y_n, η_n = len(w_grid), len(y_grid), len(η_grid)
    σ = np.empty((w_n, y_n, η_n), dtype=np.int32)
    for i in prange(w_n):
        for j in prange(y_n):
            for k in prange(η_n):
                _tmp = np.array([B(i, j, k, l, v, model) for l
                                in range(w_n)])
                σ[i, j, k] = np.argmax(_tmp)
    return σ


def optimistic_policy_iteration(model, tolerance=1e-5, m=100):
    """Optimistic policy iteration routine."""
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    w_n, y_n, η_n = len(w_grid), len(y_grid), len(η_grid)
    v = np.zeros((w_n, y_n, η_n))
    error = tolerance + 1
    while error > tolerance:
        last_v = v
        σ = get_greedy(v, model)
        for i in range(m):
            v = T_σ(v, σ, model)
        error = np.max(np.abs(v - last_v))
        print(f"OPI current error = {error}")
    return get_greedy(v, model)


## == Functions for modified OPI == ##


@njit
def D(i, j, k, l, g, model):
    """D(w, y, η, w′, g) = u(w + y - w′/η) + β g(y, w′)."""
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    w, y, η, w_1 = w_grid[i], y_grid[j], η_grid[k], w_grid[l]
    c = w + y - (w_1 / η)
    return U(c, γ) + β * g[j, l] if c > 0 else -np.inf


@njit(parallel=True)
def get_g_greedy(g, model):
    """Compute a g-greedy policy."""
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    w_n, y_n, η_n = len(w_grid), len(y_grid), len(η_grid)
    σ = np.empty((w_n, y_n, η_n), dtype=np.int32)
    for i in prange(w_n):
        for j in prange(y_n):
            for k in prange(η_n):
                _tmp = np.array([D(i, j, k, l, g, model) for l
                                in range(w_n)])
                σ[i, j, k] = np.argmax(_tmp)
    return σ


@njit(parallel=True)
def R_σ(g, σ, model):
    """The modified policy operator."""
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    w_n, y_n, η_n = len(w_grid), len(y_grid), len(η_grid)
    g_new = np.empty_like(g)
    for j in prange(y_n):
        for i_1 in prange(w_n):
            out = 0.0
            for j_1 in prange(y_n):
                for k_1 in prange(η_n):
                    out += D(i_1, j_1, k_1, σ[i_1, j_1, k_1], g,
                             model) * Q[j, j_1] * φ[k_1]
            g_new[j, i_1] = out
    return g_new


def mod_opi(model, tolerance=1e-5, m=100):
    """Modified optimistic policy iteration routine."""
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    g = np.zeros((len(y_grid), len(w_grid)))
    error = tolerance + 1
    while error > tolerance:
        last_g = g
        σ = get_g_greedy(g, model)
        for i in range(m):
            g = R_σ(g, σ, model)
        error = np.max(np.abs(g - last_g))
        print(f"OPI current error = {error}")
    return get_g_greedy(g, model)


def simulate_wealth(m):

    model = create_savings_model()
    σ_star = mod_opi(model)
    β, γ, η_grid, φ, w_grid, y_grid, Q = model

    # Simulate labor income
    mc = MarkovChain(Q)
    y_idx_series = mc.simulate(ts_length=m)

    # IID Markov chain with uniform draws
    l = len(η_grid)
    mc = MarkovChain(np.ones((l, l)) / l)
    η_idx_series = mc.simulate(ts_length=m)

    w_idx_series = np.empty_like(y_idx_series)
    w_idx_series[0] = 1  # initial condition
    for t in range(m-1):
        i, j, k = w_idx_series[t], y_idx_series[t], η_idx_series[t]
        w_idx_series[t+1] = σ_star[i, j, k]
    w_series = w_grid[w_idx_series]

    return w_series

def lorenz(v):  # assumed sorted vector
    S = np.cumsum(v)  # cumulative sums: [v[1], v[1] + v[2], ... ]
    F = np.arange(1, len(v) + 1) / len(v)
    L = S / S[-1]
    return (F, L) # returns named tuple

gini = lambda v: (2 * sum(i * y for (i, y) in enumerate(v))/sum(v) - (len(v) + 1))/len(v)

# Plots


def plot_contours(savefig=False,
                  figname="../figures_py/modified_opt_savings_1.png"):

    model = create_savings_model()
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    σ_star = optimistic_policy_iteration(model)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    y_n, η_n = len(y_grid), len(η_grid)
    y_idx, η_idx = np.arange(y_n), np.arange(η_n)
    H = np.zeros((y_n, η_n))

    w_indices = (0, len(w_grid)-1)
    titles = "low wealth", "high wealth"
    for (ax, w_idx, title) in zip(axes, w_indices, titles):

        for i_y in y_idx:
            for i_η in η_idx:
                w, y, η = w_grid[w_idx], y_grid[i_y], η_grid[i_η]
                H[i_y, i_η] = w_grid[σ_star[w_idx, i_y, i_η]] / (w + y)

        cs1 = ax.contourf(y_grid, η_grid, np.transpose(H), alpha=0.5)

        plt.colorbar(cs1, ax=ax) #, format="%.6f")

        ax.set_title(title)
        ax.set_xlabel(r"$y$")
        ax.set_ylabel(r"$\varepsilon$")

    plt.tight_layout()
    if savefig:
        fig.savefig(figname)
    plt.show()


def plot_policies(savefig=False):
    model = create_savings_model()
    β, γ, η_grid, φ, w_grid, y_grid, Q = model
    σ_star = mod_opi(model)
    y_bar = floor(len(y_grid) / 2) # index of mid-point of y_grid

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, w_grid, "k--", label=r"$45$")

    for (i, η) in enumerate(η_grid):
        label = r"$\sigma^*$" + " at " + r"$\eta = $" + f"{η.round(2)}"
        ax.plot(w_grid, w_grid[σ_star[:, y_bar, i]], label=label)
        
    ax.legend()
    plt.show()
    if savefig:
        fig.savefig("../figures_py/modified_opt_saving_2.png")


def plot_time_series(m=2_000, savefig=False):

    w_series = simulate_wealth(m)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_series, label=r"$w_t$")
    ax.set_xlabel("time")
    ax.legend()
    plt.show()
    if savefig:
        fig.savefig("../figures_py/modified_opt_saving_ts.png")


def plot_histogram(m=1_000_000, savefig=False):

    w_series = simulate_wealth(m)
    w_series.sort()
    g = round(gini(w_series), ndigits=2)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(w_series, bins=40, density=True)
    ax.set_xlabel("wealth")
    ax.text(15, 0.4, f"Gini = {g}")
    plt.show()

    if savefig:
        fig.savefig("../figures_py/modified_opt_saving_hist.png")


def plot_lorenz(m=1_000_000, savefig=False):

    w_series = simulate_wealth(m)
    w_series.sort()
    (F, L) = lorenz(w_series)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(F, F, label="Lorenz curve, equality")
    ax.plot(F, L, label="Lorenz curve, wealth distribution")
    ax.legend()
    plt.show()

    if savefig:
        fig.savefig("../figures_py/modified_opt_saving_lorenz.png")

plot_contours(savefig=True)
plot_histogram(savefig=True)
plot_lorenz(savefig=True)
plot_policies(savefig=True)
plot_time_series(savefig=True)
