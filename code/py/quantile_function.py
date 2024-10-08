from scipy.stats import rv_discrete
import numpy as np
from numba import njit


@njit
def quantile(τ, v, ϕ):
    """Compute the τ-th quantile of v(X) when X ∼ ϕ and v = sort(v)."""
    for (i, v_value) in enumerate(v):
        p = sum(ϕ[:i+1])  # sum all ϕ[j] s.t. v[j] ≤ v_value
        if p >= τ:         # exit and return v_value if prob ≥ τ
            return v_value


def R(τ, v, P):
    """For each i, compute the τ-th quantile of v(Y) when Y ∼ P(i, ⋅)"""
    return np.array([quantile(τ, v, P[i, :]) for i in range(len(v))])


def quantile_test(τ):
    ϕ = [0.1, 0.2, 0.7]
    v = [10, 20, 30]

    #d = DiscreteNonParametric(v, ϕ)
    d = rv_discrete(values=(v, ϕ))
    return quantile(τ, v, ϕ), d.ppf(τ)
    


