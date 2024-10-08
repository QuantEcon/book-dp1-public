"""

Continuous time job search.

The job status is

    s = 0 for unemployed and s = 1 for employed

The policy function has the form 

    σ[j] = optimal choice in when s = 0

We use σ[j] = 0 for reject and 1 for accept

"""

import numpy as np
import matplotlib.pyplot as plt
from quantecon.markov import tauchen
from collections import namedtuple
from numba import njit, prange
import time

# Model
Model = namedtuple("Model", ("n", "w_vals", "P", "Q", "δ", "κ", "c", "α"))

def create_js_model(α=0.1, κ=1.0, δ=0.1, n=100, ρ=0.9, ν=0.2, c=1.0):
    mc = tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P

    def Π(s, j, a, s_prime, j_prime):
        # a -= 1
        if s == 0 and s_prime == 0:
            return P[j, j_prime] * (1 - a)
        elif s == 0 and s_prime == 1:
            return P[j, j_prime] * a
        elif s == 1 and s_prime == 0:
            return P[j, j_prime]
        else:
            return 0.0

    Q = np.zeros((2, n, 2, 2, n))
    for s, j, a, s_prime, j_prime in np.ndindex(Q.shape):
        λ = κ if s == 0 else α
        Q[s, j, a, s_prime, j_prime] = λ * (Π(s, j, a, s_prime, j_prime)
                                            - (s == s_prime and j == j_prime))

    return Model(n=n, w_vals=w_vals, P=P, Q=Q, δ=δ,
                 κ=κ, c=c, α=α)

@njit
def B(s, j, a, v, model):
    n, w_vals, P, Q, δ, κ, c, α = model
    r = c if s == 0 else w_vals[j]
    continuation_value = 0
    for s_prime, j_prime in np.ndindex(2, n):
        continuation_value += v[s_prime, j_prime] * Q[s, j, a, s_prime, j_prime]
    return r + continuation_value


@njit(parallel=True)
def get_greedy(v, model):
    n = model.n
    σ = np.zeros(n, dtype=np.int32)
    for j in prange(n):
        σ[j] = np.argmax(np.array([B(0, j, a, v, model)
                                   for a in range(2)]))
    return σ


@njit(parallel=True)
def get_value(σ, model):
    n, w_vals, P, Q, δ, κ, c, α = model
    A = np.zeros((2 * n, 2 * n))
    r_σ = np.zeros(2 * n)
    for s, j in np.ndindex(2, n):
        r_σ[s * n + j] = c if s == 0 else w_vals[j]

    for s, j, s_prime, j_prime in np.ndindex(2, n, 2, n):
        A[s * n + j, s_prime * n + j_prime] = δ * (s == s_prime and j == j_prime) - Q[s, j, σ[j], s_prime, j_prime]

    v_σ = np.linalg.solve(A, r_σ)
    return v_σ.reshape(2, n)


@njit
def policy_iteration(v_init, model, tolerance=1e-9, max_iter=1000, verbose=False):
    v = v_init
    error = tolerance + 1
    k = 1
    while error > tolerance and k < max_iter:
        last_v = v
        σ = get_greedy(v, model)
        v = get_value(σ, model)
        error = np.max(np.abs(v - last_v))
        if verbose:
            print(f"Completed iteration {k}.")
        k += 1
    return v, get_greedy(v, model)


def plot_policy(savefig=False, figname="../figures_py/cont_time_js_pol.png"):
    model = create_js_model()
    n, w_vals = model.n, model.w_vals
    v_init = np.ones((2, n))
    start = time.time()
    v_star, σ_star = policy_iteration(v_init, model, verbose=True)
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(w_vals, σ_star)
    ax.set_xlabel("wage offer", fontsize=14)
    ax.set_yticks((0, 1))
    ax.set_ylabel("action (reject/accept)", fontsize=14)
    if savefig:
        plt.savefig(figname)
    plt.show()


def plot_reswage(savefig=False, figname="../figures_py/cont_time_js_res.png"):
    α_vals = np.linspace(0.05, 1.0, 100)
    res_wages_alpha = []
    for α in α_vals:
        model = create_js_model(α=α)
        n, w_vals = model.n, model.w_vals
        v_init = np.ones((2, n))
        v_star, σ_star = policy_iteration(v_init, model)
        w_idx = np.searchsorted(σ_star, 1)
        w_bar = w_vals[w_idx]
        res_wages_alpha.append(w_bar)

    κ_vals = np.linspace(0.5, 1.5, 100)
    res_wages_kappa = []
    for κ in κ_vals:
        model = create_js_model(κ=κ)
        n, w_vals = model.n, model.w_vals
        v_init = np.ones((2, n))
        v_star, σ_star = policy_iteration(v_init, model)
        w_idx = np.searchsorted(σ_star, 1)
        w_bar = w_vals[w_idx]
        res_wages_kappa.append(w_bar)

    δ_vals = np.linspace(0.05, 1.0, 100)
    res_wages_delta = []
    for δ in δ_vals:
        model = create_js_model(δ=δ)
        v_init = np.ones((2, n))
        v_star, σ_star = policy_iteration(v_init, model)
        w_idx = np.searchsorted(σ_star, 1)
        w_bar = w_vals[w_idx]
        res_wages_delta.append(w_bar)

    c_vals = np.linspace(0.5, 1.5, 100)
    res_wages_c = []
    for c in c_vals:
        model = create_js_model(c=c)
        v_init = np.ones((2, n))
        v_star, σ_star = policy_iteration(v_init, model)
        w_idx = np.searchsorted(σ_star, 1)
        w_bar = w_vals[w_idx]
        res_wages_c.append(w_bar)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5))

    ax = axes[0, 0]
    ax.plot(α_vals, res_wages_alpha)
    ax.set_xlabel("separation rate", fontsize=14)
    ax.set_ylabel("res. wage", fontsize=14)

    ax = axes[0, 1]
    ax.plot(κ_vals, res_wages_kappa)
    ax.set_xlabel("offer rate", fontsize=14)
    ax.set_ylabel("res. wage", fontsize=14)

    ax = axes[1, 0]
    ax.plot(δ_vals, res_wages_delta)
    ax.set_xlabel("discount rate", fontsize=14)
    ax.set_ylabel("res. wage", fontsize=14)

    ax = axes[1, 1]
    ax.plot(c_vals, res_wages_c)
    ax.set_xlabel("unempl. compensation", fontsize=14)
    ax.set_ylabel("res. wage", fontsize=14)

    fig.tight_layout()

    if savefig:
        plt.savefig(figname)
    plt.show()
