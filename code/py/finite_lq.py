from quantecon import compute_fixed_point
from quantecon.markov import tauchen, MarkovChain

import numpy as np
from collections import namedtuple
from numba import njit, prange
import time


# NamedTuple Model
Model = namedtuple("Model", ("β", "a_0", "a_1", "γ", "c",
                             "y_grid", "z_grid", "Q"))


def create_investment_model(
        r=0.04,                               # Interest rate
        a_0=10.0, a_1=1.0,                    # Demand parameters
        γ=25.0, c=1.0,                        # Adjustment and unit cost
        y_min=0.0, y_max=20.0, y_size=100,    # Grid for output
        ρ=0.9, ν=1.0,                         # AR(1) parameters
        z_size=25):                           # Grid size for shock
    β = 1/(1+r)
    y_grid = np.linspace(y_min, y_max, y_size)
    mc = tauchen(y_size, ρ, ν)
    z_grid, Q = mc.state_values, mc.P
    return Model(β=β, a_0=a_0, a_1=a_1, γ=γ, c=c,
          y_grid=y_grid, z_grid=z_grid, Q=Q)


@njit
def B(i, j, k, v, model):
    """
    The aggregator B is given by

        B(y, z, y′) = r(y, z, y′) + β Σ_z′ v(y′, z′) Q(z, z′)."

    where

        r(y, z, y′) := (a_0 - a_1 * y + z - c) y - γ * (y′ - y)^2

    """
    β, a_0, a_1, γ, c, y_grid, z_grid, Q = model
    y, z, y_1 = y_grid[i], z_grid[j], y_grid[k]
    r = (a_0 - a_1 * y + z - c) * y - γ * (y_1 - y)**2
    return r + β * np.dot(v[k, :], Q[j, :])


@njit(parallel=True)
def T_σ(v, σ, model):
    """The policy operator."""
    v_new = np.empty_like(v)
    for i in prange(len(model.y_grid)):
        for j in prange(len(model.z_grid)):
            v_new[i, j] = B(i, j, σ[i, j], v, model)
    return v_new


@njit(parallel=True)
def T(v, model):
    """The Bellman operator."""
    v_new = np.empty_like(v)
    for i in prange(len(model.y_grid)):
        for j in prange(len(model.z_grid)):
            tmp = np.array([B(i, j, k, v, model) for k
                            in np.arange(len(model.y_grid))])
            v_new[i, j] = np.max(tmp)
    return v_new


@njit(parallel=True)
def get_greedy(v, model):
    """Compute a v-greedy policy."""
    n, m = len(model.y_grid), len(model.z_grid)
    σ = np.empty((n, m), dtype=np.int32)
    for i in prange(n):
        for j in prange(m):
            tmp = np.array([B(i, j, k, v, model) for k
                            in np.arange(n)])
            σ[i, j] = np.argmax(tmp)
    return σ



def value_iteration(model, tol=1e-5):
    """Value function iteration routine."""
    vz = np.zeros((len(model.y_grid), len(model.z_grid)))
    v_star = compute_fixed_point(lambda v: T(v, model), vz,
                                 error_tol=tol, max_iter=1000, print_skip=25)
    return get_greedy(v_star, model)


@njit
def single_to_multi(m, zn):
    # Function to extract (i, j) from m = i + (j-1)*zn
    return (m//zn, m%zn)


@njit(parallel=True)
def get_value(σ, model):
    """Get the value v_σ of policy σ."""
    # Unpack and set up
    β, a_0, a_1, γ, c, y_grid, z_grid, Q = model
    yn, zn = len(y_grid), len(z_grid)
    n = yn * zn
    # Allocate and create single index versions of P_σ and r_σ
    P_σ = np.zeros((n, n))
    r_σ = np.zeros(n)
    for m in prange(n):
        i, j = single_to_multi(m, zn)
        y, z, y_1 = y_grid[i], z_grid[j], y_grid[σ[i, j]]
        r_σ[m] = (a_0 - a_1 * y + z - c) * y - γ * (y_1 - y)**2
        for m_1 in prange(n):
            i_1, j_1 = single_to_multi(m_1, zn)
            if i_1 == σ[i, j]:
                P_σ[m, m_1] = Q[j, j_1]

    I = np.identity(n)
    # Solve for the value of σ
    v_σ = np.linalg.solve((I - β * P_σ), r_σ)
    # Return as multi-index array
    v_σ = v_σ.reshape(yn, zn)
    return v_σ


@njit
def policy_iteration(model):
    """Howard policy iteration routine."""
    yn, zn = len(model.y_grid), len(model.z_grid)
    σ = np.ones((yn, zn), dtype=np.int32)
    i, error = 0, 1.0
    while error > 0:
        v_σ = get_value(σ, model)
        σ_new = get_greedy(v_σ, model)
        error = np.max(np.abs(σ_new - σ))
        σ = σ_new
        i = i + 1
        print(f"Concluded loop {i} with error: {error}.")
    return σ


@njit
def optimistic_policy_iteration(model, tol=1e-5, m=100):
    """Optimistic policy iteration routine."""
    v = np.zeros((len(model.y_grid), len(model.z_grid)))
    error = tol + 1
    while error > tol:
        last_v = v
        σ = get_greedy(v, model)
        for i in range(m):
            v = T_σ(v, σ, model)
        error = np.max(np.abs(v - last_v))
    return get_greedy(v, model)


# Plots

import matplotlib.pyplot as plt


def plot_policy(savefig=False, figname="./figures/finite_lq_0.pdf"):
    model = create_investment_model()
    β, a_0, a_1, γ, c, y_grid, z_grid, Q = model
    σ_star = optimistic_policy_iteration(model)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(y_grid, y_grid, "k--", label=r"$45$")
    ax.plot(y_grid, y_grid[σ_star[:, 0]], label=r"$\sigma^*(\cdot, z_1)$")
    ax.plot(y_grid, y_grid[σ_star[:, -1]], label="$\sigma^*(\cdot, z_N)$")
    ax.legend()
    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_sim(savefig=False, figname="./figures/finite_lq_1.pdf"):
    ts_length = 200

    fig, axes = plt.subplots(4, 1, figsize=(9, 11.2))

    for (ax, γ) in zip(axes, (1, 10, 20, 30)):
        model = create_investment_model(γ=γ)
        β, a_0, a_1, γ, c, y_grid, z_grid, Q = model
        σ_star = optimistic_policy_iteration(model)
        mc = MarkovChain(Q, z_grid)

        z_sim_idx = mc.simulate_indices(ts_length)
        z_sim = z_grid[z_sim_idx]

        y_sim_idx = np.empty(ts_length, dtype=np.int32)
        y_1 = (a_0 - c + z_sim[1]) / (2 * a_1)

        y_sim_idx[0] = np.searchsorted(y_grid, y_1)
        for t in range(ts_length-1):
            y_sim_idx[t+1] = σ_star[y_sim_idx[t], z_sim_idx[t]]
        y_sim = y_grid[y_sim_idx]
        y_bar_sim = (a_0 - c + z_sim) / (2 * a_1)

        ax.plot(np.arange(1, ts_length+1), y_sim, label=r"$Y_t$")
        ax.plot(np.arange(1, ts_length+1), y_bar_sim, label=r"$\bar Y_t$")
        ax.legend(frameon=False, loc="upper right")
        ax.set_ylabel("output")
        ax.set_ylim(1, 9)
        ax.set_title(r"$\gamma = $" + f"{γ}")

    fig.tight_layout()
    plt.show()
    if savefig:
        fig.savefig(figname)


def plot_timing(m_vals=np.arange(1, 601, 10),
                savefig=False,
                figname="./figures/finite_lq_time.pdf"
    ):
    # NOTE: Uncomment the following lines in this function to
    # include Policy iteration plot
    model = create_investment_model()
    # print("Running Howard policy iteration.")
    # t1 = time.time()
    # σ_pi = policy_iteration(model)
    # pi_time = time.time() - t1
    # print(f"PI completed in {pi_time} seconds.")
    print("Running value function iteration.")
    t1 = time.time()
    σ_vfi = value_iteration(model)
    vfi_time = time.time() - t1
    print(f"VFI completed in {vfi_time} seconds.")
    opi_times = []
    for m in m_vals:
        print(f"Running optimistic policy iteration with m={m}.")
        t1 = time.time()
        σ_opi = optimistic_policy_iteration(model, m=m, tol=1e-5)
        t2 = time.time()
        print(f"OPI with m={m} completed in {t2-t1} seconds.")
        opi_times.append(t2-t1)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(m_vals, [vfi_time]*len(m_vals),
            linewidth=2, label="value function iteration")
    # ax.plot(m_vals, [pi_time]*len(m_vals),
    #         linewidth=2, label="Howard policy iteration")
    ax.plot(m_vals, opi_times, linewidth=2, label="optimistic policy iteration")
    ax.legend(frameon=False)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel("time")
    plt.show()
    if savefig:
        fig.savefig(figname)
    return (vfi_time, opi_times)
