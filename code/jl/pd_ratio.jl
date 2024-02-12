"""
Price-dividend ratio in a model with dividend and consumption growth.

"""

using QuantEcon, LinearAlgebra

"Creates an instance of the asset pricing model with Markov state."
function create_asset_pricing_model(;
        n=200,              # state grid size
        ρ=0.9, ν=0.2,       # state persistence and volatility
        β=0.99, γ=2.5,      # discount and preference parameter
        μ_c=0.01, σ_c=0.02, # consumption growth mean and volatility
        μ_d=0.02, σ_d=0.1)  # dividend growth mean and volatility
    mc = tauchen(n, ρ, ν)
    x_vals, P = exp.(mc.state_values), mc.p
    return (; x_vals, P, β, γ, μ_c, σ_c, μ_d, σ_d)
end

" Build the discount matrix A. "
function build_discount_matrix(model)
    (; x_vals, P, β, γ, μ_c, σ_c, μ_d, σ_d) = model
    e = exp.(μ_d - γ*μ_c + (γ^2*σ_c^2 + σ_d^2)/2 .+ (1-γ)*x_vals)
    return β * e .* P
end

"Compute the price-dividend ratio associated with the model."
function pd_ratio(model)
    (; x_vals, P, β, γ, μ_c, σ_c, μ_d, σ_d) = model
    A = build_discount_matrix(model)
    @assert maximum(abs.(eigvals(A))) < 1 "Requires r(A) < 1."
    n = length(x_vals)
    return (I - A) \ (A * ones(n))
end


# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

default_model = create_asset_pricing_model()


function plot_main(; μ_d_vals = (0.02, 0.08),
                     savefig=false, 
                     figname="./figures/pd_ratio_1.pdf")
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for μ_d in μ_d_vals
        model = create_asset_pricing_model(μ_d=μ_d)
        (; x_vals, P, β, γ, μ_c, σ_c, μ_d, σ_d) = model
        v_star = pd_ratio(model)
        ax.plot(x_vals, v_star, lw=2, alpha=0.6, label=L"\mu_d="*"$μ_d")
    end

    ax.legend(frameon=false, fontsize=fontsize)
    ax.set_xlabel(L"x", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

