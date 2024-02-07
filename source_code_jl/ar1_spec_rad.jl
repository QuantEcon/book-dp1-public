""" 

Compute r(L) for model

    Zₜ = μ (1 - ρ) + ρ Zₜ₋₁ + σ εₜ
    β_t = b(Z_t)
    
The process is discretized using the Tauchen method with n states.
"""

using LinearAlgebra, QuantEcon

function compute_mc_spec_rad(n, ρ, σ, μ, m, b)
    mc = tauchen(n, ρ, σ, μ * (1 - ρ), m)
    state_values, P = mc.state_values, mc.p

    L = zeros(n, n)
    for i in 1:n
        for j in 1:n
            L[i, j] = b(state_values[i]) * P[i, j]
        end
    end
    r = maximum(abs.(eigvals(L)))
    return r
end
    
# Hubmer et al parameter values, p. 24 of May 17 2020 version.

n = 15
ρ = 0.992
σ = 0.0006
μ = 0.944 
m = 4
b(z) = z

println("Spectral radius of L in Hubmer et al.:")
println(compute_mc_spec_rad(n, ρ, σ, μ, m, b))

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
b(z) = beta * z

println("Spectral radius of L in Hills et al.:")
println(compute_mc_spec_rad(n, ρ, σ, μ, m, b))

# Let's run a simulation of the discount process.
# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fs=14

function plot_beta_sim(; T=80,
                         savefig=false, 
                         figname="./figures/ar1_spec_rad.pdf")
    β_vals = zeros(T)
    Z = 1
    for t in 1:T
        β_vals[t] = beta * Z
        Z = 1 - ρ + ρ * Z + σ * randn()
    end
        
    fig, ax = plt.subplots(figsize=(6, 3.8))

    ax.plot(β_vals, label=L"\beta_t")
    ax.plot(1:T, ones(T), "k--", alpha=0.5, label=L"\beta=1")
    ax.set_yticks((0.97, 1.0, 1.03))
    ax.set_xlabel("time")
    ax.legend(frameon=false, fontsize=fs)

    if savefig
        fig.savefig(figname)
    end
    plt.show()
end
