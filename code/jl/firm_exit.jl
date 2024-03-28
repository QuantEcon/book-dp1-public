"""
Firm valuation with exit option.

"""

using QuantEcon, LinearAlgebra
include("s_approx.jl")

"Creates an instance of the firm exit model."
function create_exit_model(;
        n=200,                  # productivity grid size
        ρ=0.95, μ=0.1, ν=0.1,   # persistence, mean and volatility
        β=0.98, s=100.0         # discount factor and scrap value
    )
    mc = tauchen(n, ρ, ν, μ)
    z_vals, Q = mc.state_values, mc.p
    return (; n, z_vals, Q, β, s)
end


"Compute value of firm without exit option."
function no_exit_value(model)
    (; n, z_vals, Q, β, s) = model
    return (I - β * Q) \ z_vals
end

" The Bellman operator Tv = max{s, π + β Q v}."
function T(v, model)
    (; n, z_vals, Q, β, s) = model
    h = z_vals .+ β * Q * v
    return max.(s, h)
end

" Get a v-greedy policy."
function get_greedy(v, model)
    (; n, z_vals, Q, β, s) = model
    σ = s .>= z_vals .+ β * Q * v
    return σ
end

"Solve by VFI."
function vfi(model) 
    v_init = no_exit_value(model)
    v_star = successive_approx(v -> T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
end


# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16


function plot_val(; savefig=false, 
                     figname="../figures/firm_exit_1.pdf")

    fig, ax = plt.subplots(figsize=(9, 5.2))

    model = create_exit_model()
    (; n, z_vals, Q, β, s) = model

    v_star, σ_star = vfi(model)
    h = z_vals + β * Q * v_star

    ax.plot(z_vals, h, "-", lw=3, alpha=0.6, label=L"h^*")
    ax.plot(z_vals, s * ones(n), "-", lw=3, alpha=0.6, label=L"s")
    ax.plot(z_vals, v_star, "k--", lw=1.5, alpha=0.8, label=L"v^*")

    ax.legend(frameon=false, fontsize=fontsize)
    ax.set_xlabel(L"z", fontsize=fontsize)

    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


function plot_comparison(; savefig=false, 
                     figname="../figures/firm_exit_2.pdf")

    fig, ax = plt.subplots(figsize=(9, 5.2))

    model = create_exit_model()
    (; n, z_vals, Q, β, s) = model

    v_star, σ_star = vfi(model)
    w = no_exit_value(model)

    ax.plot(z_vals, v_star, "k-", lw=2, alpha=0.6, label=L"v^*")
    ax.plot(z_vals, w, lw=2, alpha=0.6, label="no-exit value")

    ax.legend(frameon=false, fontsize=fontsize)
    ax.set_xlabel(L"z", fontsize=fontsize)

    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


