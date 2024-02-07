from quantecon.markov import tauchen
import numpy as np
from collections import namedtuple
from numba import njit, prange


# NamedTuple Model
Model = namedtuple("Model", ("β", "R", "γ", "w_grid", "y_grid", "Q"))


def create_savings_model(R=1.01, β=0.98, γ=2.5,
                         w_min=0.01, w_max=20.0, w_size=200,
                         ρ=0.9, ν=0.1, y_size=5):
    w_grid = np.linspace(w_min, w_max, w_size)
    mc = tauchen(y_size, ρ, ν)
    y_grid, Q = np.exp(mc.state_values), mc.P
    return Model(β=β, R=R, γ=γ, w_grid=w_grid, y_grid=y_grid, Q=Q)


@njit
def U(c, γ):
    return c**(1-γ)/(1-γ)


@njit
def B(i, j, k, v, model):
    """
    B(w, y, w′, v) = u(R*w + y - w′) + β Σ_y′ v(w′, y′) Q(y, y′).
    """
    β, R, γ, w_grid, y_grid, Q = model
    w, y, w_1 = w_grid[i], y_grid[j], w_grid[k]
    c = w + y - (w_1 / R)
    value = -np.inf
    if c > 0:
        value = U(c, γ) + β * np.dot(v[k, :], Q[j, :])
    return value


@njit(parallel=True)
def T(v, model):
    """The Bellman operator."""
    β, R, γ, w_grid, y_grid, Q = model
    v_new = np.empty_like(v)
    for i in prange(w_grid.shape[0]):
        for j in prange(y_grid.shape[0]):
            x_tmp = np.array([B(i, j, k, v, model) for k
                              in np.arange(w_grid.shape[0])])
            v_new[i, j] = np.max(x_tmp)
    return v_new


@njit(parallel=True)
def T_σ(v, σ, model):
    """The policy operator."""
    β, R, γ, w_grid, y_grid, Q = model
    v_new = np.empty_like(v)
    for i in prange(w_grid.shape[0]):
        for j in prange(y_grid.shape[0]):
            v_new[i, j] = B(i, j, σ[i, j], v, model)
    return v_new
