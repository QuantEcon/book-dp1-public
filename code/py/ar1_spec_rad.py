"""

Compute r(L) for model

    Zₜ = μ (1 - ρ) + ρ Zₜ₋₁ + σ εₜ
    β_t = b(Z_t)

The process is discretized using the Tauchen method with n states.
"""

import numpy as np
import matplotlib.pyplot as plt

from quantecon.markov import tauchen


def compute_mc_spec_rad(n, ρ, σ, μ, m, b):
    mc = tauchen(n, ρ, σ, μ * (1 - ρ), m)
    state_values, P = mc.state_values, mc.P

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i, j] = b(state_values[i]) * P[i, j]
    r = np.max(np.abs(np.linalg.eigvals(L)))
    return r


# Hubmer et al parameter values, p. 24 of May 17 2020 version.

n = 15
ρ = 0.992
σ = 0.0006
μ = 0.944
m = 4
b = lambda z: z

print("Spectral radius of L in Hubmer et al.:")
print(compute_mc_spec_rad(n, ρ, σ, μ, m, b))

# ## Hills et al 2019 EER

# For the empirical model,
#
# $$
#     Z_{t+1} = 1 - \rho + \rho Z_t + \sigma \epsilon_{t+1},
#     \quad \beta_t = \beta Z_t
# $$
#
# with
#
# $$
#     \beta = 0.99875, \; \rho = 0.85, \; \sigma = 0.0062
# $$
#
# They use 15 grid points on $[1-4.5\sigma_\delta, 1+4.5\sigma_\delta]$.

n = 15
ρ = 0.85
σ = 0.0062
μ = 1
m = 4.5
beta = 0.99875
b = lambda z: beta*z

print("Spectral radius of L in Hills et al.:")
print(compute_mc_spec_rad(n, ρ, σ, μ, m, b))

# Let's run a simulation of the discount process.
# Plots


def plot_beta_sim(T=80,
                  savefig=True,
                  figname="../figures_py/ar1_spec_rad.png"):
    β_vals = np.zeros(T)
    Z = 1
    for t in range(T):
        β_vals[t] = beta * Z
        Z = 1 - ρ + ρ * Z + σ * np.random.default_rng().normal()

    fig, ax = plt.subplots(figsize=(6, 3.8))

    ax.plot(β_vals, label=r"$\beta_t$")
    ax.plot(np.arange(T), np.ones(T), "k--", alpha=0.5, label=r"$\beta=1$")
    ax.set_yticks((0.97, 1.0, 1.03))
    ax.set_xlabel("time")
    ax.legend(frameon=False)

    if savefig:
        fig.savefig(figname)
    plt.show()
