import numpy as np
from scipy.stats import binom
from numba import njit, prange
from collections import namedtuple

Model = namedtuple('Model', ['α', 'β', 'γ', 'θ', 'φ', 'e_grid', 'w_grid'])

def create_ez_model(ψ=1.97,
                    β=0.96,
                    γ=-7.89,
                    n=80,
                    p=0.5,
                    e_max=0.5,
                    w_size=50,
                    w_max=2):
    α = 1 - 1/ψ
    θ = γ / α
    b = binom(n - 1, p)
    φ = b.pmf(np.arange(n))
    e_grid = np.linspace(1e-5, e_max, n)
    w_grid = np.linspace(0, w_max, w_size)
    return Model(α=α, β=β, γ=γ, θ=θ, φ=φ,
                 e_grid=e_grid, w_grid=w_grid)

@njit
def B(i, j, k, v, model):
    """
    Action-value aggregator for the original model.
    """
    α, β, γ, θ, φ, e_grid, w_grid = model
    w, e, s = w_grid[i], e_grid[j], w_grid[k]
    if s<= w:
        Rv = np.dot(v[k, :]**γ, φ)**(1/γ)
        return ((w - s + e)**α + β * Rv**α)**(1/α)
    return -np.inf

@njit
def B2(i, k, h, model):
    α, β, γ, θ, φ, e_grid, w_grid = model
    w, s = w_grid[i], w_grid[k]
    if s <= w:
        Ge = ((w - s + e_grid)**α + β * h[k]**α)**(1/α)
        return np.dot(Ge**γ, φ)**(1/γ)
    return -np.inf


@njit 
def G_obj(i, j, k, h, model):
    α, β, γ, θ, φ, e_grid, w_grid = model
    w, e, s = w_grid[i], e_grid[j], w_grid[k]
    if s <= w:
        return ((w - s + e)**α + β * h[k]**α)**(1/α)
    return -np.inf


@njit
def G_max(h, model):
    w_n, e_n = len(model.w_grid), len(model.e_grid)
    σ_star_mod = np.zeros((w_n, e_n), dtype=np.int32)
    for i in prange(w_n):
        for j in range(e_n):
            G_values = np.array([G_obj(i, j, k, h, model)
                                 for k in range(w_n)])
            σ_star_mod[i, j] = np.argmax(G_values)
    return σ_star_mod
