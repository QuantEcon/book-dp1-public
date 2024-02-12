"""
Epstein--Zin utility: solving the recursion for a given consumption
path.

"""

include("s_approx.jl")
using LinearAlgebra, QuantEcon

function create_ez_utility_model(;
        n=200,      # size of state space
        ρ=0.96,     # correlation coef in AR(1)
        σ=0.1,      # volatility
        β=0.99,     # time discount factor
        α=0.75,     # EIS parameter
        γ=-2.0)     # risk aversion parameter

    mc = tauchen(n, ρ, σ, 0, 5) 
    x_vals, P = mc.state_values, mc.p 
    c = exp.(x_vals)      

    return (; β, ρ, σ, α, γ, c, x_vals, P)
end

function K(v, model)
    (; β, ρ, σ, α, γ, c, x_vals, P) = model

    R = (P * (v.^γ)).^(1/γ)
    return ((1 - β) * c.^α + β * R.^α).^(1/α)
end

function compute_ez_utility(model)
    v_init = ones(length(model.x_vals))
    v_star = successive_approx(v -> K(v, model), 
                               v_init, 
                               tolerance=1e-10)
    return v_star
end


# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

function plot_convergence(; savefig=false, 
                  num_iter=100,
                  figname="./figures/ez_utility_c.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    model = create_ez_utility_model()
    (; β, ρ, σ, α, γ, c, x_vals, P) = model


    v_star = compute_ez_utility(model)
    v = 0.1 * v_star
    ax.plot(x_vals, v, lw=3, "k-", alpha=0.7, label=L"v_0")

    greys = [string(g) for g in LinRange(0.0, 0.4, num_iter)]
    greys = reverse(greys)

    for (i, g) in enumerate(greys)
        ax.plot(x_vals, v, "k-", color=g, lw=1, alpha=0.7)
        for t in 1:20
            v = K(v, model)
        end
    end

    v_star = compute_ez_utility(model)
    ax.plot(x_vals, v_star, lw=3, alpha=0.7, label=L"v^*")
    ax.set_xlabel(L"x", fontsize=fontsize)

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


function plot_v(; savefig=false, 
                  figname="./figures/ez_utility_1.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    model = create_ez_utility_model()
    (; β, ρ, σ, α, γ, c, x_vals, P) = model
    v_star = compute_ez_utility(model)
    ax.plot(x_vals, v_star, lw=2, alpha=0.7, label=L"v^*")
    ax.set_xlabel(L"x", fontsize=fontsize)

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


function vary_gamma(; gamma_vals=[1.0, -8.0],
                  savefig=false, 
                  figname="./figures/ez_utility_2.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    
    for γ in gamma_vals
        model = create_ez_utility_model(γ=γ)
        (; β, ρ, σ, α, γ, c, x_vals, P) = model
        v_star = compute_ez_utility(model)
        ax.plot(x_vals, v_star, lw=2, alpha=0.7, label=L"\gamma="*"$γ")
        ax.set_xlabel(L"x", fontsize=fontsize)
        ax.set_ylabel(L"v(x)", fontsize=fontsize)
    end

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


function vary_alpha(; alpha_vals=[0.5, 0.6],
                  savefig=false, 
                  figname="./figures/ez_utility_3.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    
    for α in alpha_vals
        model = create_ez_utility_model(α=α)
        (; β, ρ, σ, α, γ, c, x_vals, P) = model
        v_star = compute_ez_utility(model)
        ax.plot(x_vals, v_star, lw=2, alpha=0.7, label=L"\alpha="*"$α")
        ax.set_xlabel(L"x", fontsize=fontsize)
        ax.set_ylabel(L"v(x)", fontsize=fontsize)
    end

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

