import numpy as np
from numba import njit, prange
from ez_model import B, B2


@njit(parallel=True)
def T_σ(v, σ, model):
    w_n, e_n = v.shape
    v_new = np.zeros_like(v)
    for i in prange(w_n):
        for j in range(e_n):
            v_new[i, j] = B(i, j, σ[i, j], v, model)
    return v_new


@njit(parallel=True)
def T2_σ(h, σ, model):
    w_n = len(h)
    h_new = np.zeros_like(h)
    for i in prange(w_n):
        h_new[i] = B2(i, σ[i], h, model)
    return h_new


@njit(parallel=True)
def get_greedy(v, model):
    w_n, e_n = v.shape
    σ = np.zeros_like(v, dtype=np.int32)
    for i in prange(w_n):
        for j in range(e_n):
            B_values = np.array([B(i, j, k, v, model) for k in range(w_n)])
            σ[i, j] = np.argmax(B_values)
    return σ


@njit(parallel=True)
def get_greedy2(h, model):
    w_n = len(h)
    σ = np.zeros(w_n, dtype=np.int32)
    for i in prange(w_n):
        B_values = np.array([B2(i, k, h, model) for k in range(w_n)])
        σ[i] = np.argmax(B_values)
    return σ


@njit
def get_value(v_init, σ, m, model):
    v = v_init
    for _ in range(m):
        v = T_σ(v, σ, model)
    return v

@njit
def get_value2(h_init, σ, m, model):
    h = h_init
    for _ in range(m):
        h = T2_σ(h, σ, model)
    return h


def optimistic_policy_iteration(v_init, model, tolerance=1e-9, max_iter=1000, m=100):
    v = v_init
    error = tolerance + 1
    k = 1
    while error > tolerance and k < max_iter:
        last_v = v
        σ = get_greedy(v, model)
        v = get_value(v, σ, m, model)
        error = np.max(np.abs(v - last_v))
        print(f"Completed iteration {k} with error {error}.")
        k += 1
    return v, get_greedy(v, model)

def optimistic_policy_iteration2(h_init, model, tolerance=1e-9, max_iter=1000, m=100):
    h = h_init
    error = tolerance + 1
    k = 1
    while error > tolerance and k < max_iter:
        last_h = h
        σ = get_greedy2(h, model)
        h = get_value2(h, σ, m, model)
        error = np.max(np.abs(h - last_h))
        print(f"Completed iteration {k} with error {error}.")
        k += 1
    return h, get_greedy2(h, model)
