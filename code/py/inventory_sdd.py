"""

Inventory management model with state-dependent discounting. The discount
factor takes the form β_t = Z_t, where (Z_t) is a discretization of a
Gaussian AR(1) process

    X_t = ρ X_{t-1} + b + ν W_t.

"""

from quantecon import compute_fixed_point
from quantecon.markov import tauchen, MarkovChain

import numpy as np
from time import time
from numba import njit, prange
from collections import namedtuple
import matplotlib.pyplot as plt

# NamedTuple Model
Model = namedtuple("Model", ("K", "c", "κ", "p", "r", 
                             "R", "y_vals", "z_vals", "Q"))


@njit
def ϕ(p, d):
    return (1 - p)**d * p

@njit
def f(y, a, d):
    return np.maximum(y - d, 0) + a  # Inventory update

def create_sdd_inventory_model(
            ρ=0.98, ν=0.002, n_z=20, b=0.97,  # Z state parameters
            K=40, c=0.2, κ=0.8, p=0.6,        # firm and demand parameters
            d_max=100):                        # truncation of demand shock
    d_vals = np.arange(d_max+1)
    ϕ_vals = ϕ(p, d_vals)
    y_vals = np.arange(K+1)
    n_y = len(y_vals)
    mc = tauchen(n_z, ρ, ν)
    z_vals, Q = mc.state_values + b, mc.P
    ρL = np.max(np.abs(np.linalg.eigvals(z_vals * Q)))
    assert ρL < 1, "Error: ρ(L) >= 1."    # check r(L) < 1

    R = np.zeros((n_y, n_y, n_y))
    for i_y, y in enumerate(y_vals):
        for i_y_1, y_1 in enumerate(y_vals):
            for i_a, a in enumerate(range(K - y + 1)):
                hits = [f(y, a, d) == y_1 for d in d_vals]
                R[i_y, i_a, i_y_1] = np.dot(hits, ϕ_vals)


    r = np.empty((n_y, n_y))
    for i_y, y in enumerate(y_vals):
        for i_a, a in enumerate(range(K - y + 1)):
            cost = c * a + κ * (a > 0)
            r[i_y, i_a] = np.dot(np.minimum(y, d_vals),  ϕ_vals) - cost


    return Model(K=K, c=c, κ=κ, p=p, r=r, R=R, 
                 y_vals=y_vals, z_vals=z_vals, Q=Q)

@njit
def B(i_y, i_z, i_a, v, model):
    """
    The function B(x, z, a, v) = r(x, a) + β(z) Σ_x′ v(x′) P(x, a, x′).
    """
    K, c, κ, p, r, R, y_vals, z_vals, Q = model
    β = z_vals[i_z]
    cv = 0.0

    for i_z_1 in prange(len(z_vals)):
        for i_y_1 in prange(len(y_vals)):
            cv += v[i_y_1, i_z_1] * R[i_y, i_a, i_y_1] * Q[i_z, i_z_1]
    return r[i_y, i_a] + β * cv

@njit(parallel=True)
def T(v, model):
    """The Bellman operator."""
    K, c, κ, p, r, R, y_vals, z_vals, Q = model
    new_v = np.empty_like(v)
    for i_z in prange(len(z_vals)):
        for (i_y, y) in enumerate(y_vals):
            Γy = np.arange(K - y + 1)
            new_v[i_y, i_z] = np.max(np.array([B(i_y, i_z, i_a, v, model) 
                               for i_a in Γy]))
    return new_v

@njit
def T_σ(v, σ, model):
    """The policy operator."""
    K, c, κ, p, r, R, y_vals, z_vals, Q = model
    new_v = np.empty_like(v)
    for (i_z, z) in enumerate(z_vals):
        for (i_y, y) in enumerate(y_vals):
            new_v[i_y, i_z] = B(i_y, i_z, σ[i_y, i_z], v, model)
    return new_v

@njit(parallel=True)
def get_greedy(v, model):
    """Get a v-greedy policy.  Returns a zero-based array."""
    K, c, κ, p, r, R, y_vals, z_vals, Q = model
    n_z = len(z_vals)
    σ_star = np.zeros((K+1, n_z), dtype=np.int32)
    for (i_z, z) in enumerate(z_vals):
        for (i_y, y) in enumerate(y_vals):
            Γy = np.arange(K - y + 1)
            i_a = np.argmax(np.array([B(i_y, i_z, i_a, v, model) 
                               for i_a in Γy]))
            σ_star[i_y, i_z] = Γy[i_a]
    return σ_star

@njit
def get_value(v_init, σ, m, model):
    """Approximate lifetime value of policy σ."""
    v = v_init
    for _ in range(m):
        v = T_σ(v, σ, model)
    return v

def solve_inventory_model(v_init, model):
    """Use successive_approx to get v_star and then compute greedy."""
    v_star = compute_fixed_point(lambda v: T(v, model), v_init,
                                 error_tol=1e-5, max_iter=1000, print_skip=25)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star

def optimistic_policy_iteration(v_init, 
                                model,
                                tolerance=1e-6, 
                                max_iter=1_000,
                                print_step=10,
                                m=60):
    v = v_init
    error = tolerance + 1
    k = 1
    while (error > tolerance) and (k < max_iter):
        last_v = v
        σ = get_greedy(v, model)
        v = get_value(v, σ, m, model)
        error = np.max(np.abs(v - last_v))
        if k % print_step == 0:
            print(f"Completed iteration {k} with error {error}.")
        k += 1
    return v, get_greedy(v, model)


# == Plots == #

# Create an instance of the model and solve it
model = create_sdd_inventory_model()
K, c, κ, p, r, R, y_vals, z_vals, Q = model
n_z = len(z_vals)
v_init = np.zeros((K+1, n_z), dtype=float)
print("Solving model.")
v_star, σ_star = optimistic_policy_iteration(v_init, model)
z_mc = MarkovChain(Q, z_vals)

def sim_inventories(ts_length, X_init=0):
    """Simulate given the optimal policy."""
    global p, z_mc
    i_z = z_mc.simulate_indices(ts_length, init=1)
    X = np.zeros(ts_length, dtype=np.int32)
    X[0] = X_init
    rand = np.random.default_rng().geometric(p=p, size=ts_length-1) - 1
    for t in range(ts_length-1):
        X[t+1] = f(X[t], σ_star[X[t], i_z[t]], rand[t])
    return X, z_vals[i_z]

def plot_ts(ts_length=400,
            fontsize=10,
            figname="../figures_py/inventory_sdd_ts.png",
            savefig=False):
    
    X, Z = sim_inventories(ts_length)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))

    ax = axes[0]
    ax.plot(X, label="inventory", alpha=0.7)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_ylim(0, np.max(X)+3)

    # calculate interest rate from discount factors
    r = (1 / Z) - 1

    ax = axes[1]
    ax.plot(r, label=r"$r_t$", alpha=0.7)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)

    plt.tight_layout()
    plt.show()
    if savefig:
        fig.savefig(figname)

def plot_timing(m_vals=np.arange(1, 400, 10),
                fontsize=16,
                savefig=False):
    print("Running value function iteration.")
    t_start = time()
    solve_inventory_model(v_init, model)
    vfi_time = time() - t_start
    print(f"VFI completed in {vfi_time} seconds.")
    opi_times = []
    for m in m_vals:
        print(f"Running optimistic policy iteration with m = {m}.")
        t_start = time()
        optimistic_policy_iteration(v_init, model, m=m)
        opi_time = time() - t_start
        print(f"OPI with m = {m} completed in {opi_time} seconds.")
        opi_times.append(opi_time)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(m_vals, np.full(len(m_vals), vfi_time),
            lw=2, label="value function iteration")
    ax.plot(m_vals, opi_times, lw=2, label="optimistic policy iteration")
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_xlabel(r"$m$", fontsize=fontsize)
    ax.set_ylabel("time", fontsize=fontsize)
    plt.show()
    if savefig:
        fig.savefig("../figures_py/inventory_sdd_timing.png")
    return (opi_time, vfi_time, opi_times)

plot_ts(savefig=True)
plot_timing(savefig=True)
