"""
Infinite-horizon job search with Markov wage draws and risk-sensitive preferences.

"""

using QuantEcon, LinearAlgebra
include("s_approx.jl")

"Creates an instance of the job search model with Markov wages."
function create_markov_js_model(;
        n=200,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.98,      # discount factor
        c=1.0,       # unemployment compensation
        θ=-0.01       # risk parameter
    )
    mc = tauchen(n, ρ, ν)
    w_vals, P = exp.(mc.state_values), mc.p
    return (; n, w_vals, P, β, c, θ)
end

"""
The Bellman operator Tv = max{e, c + β R v} with 

    e(w) = w / (1-β) and

    (Rv)(w) = (1/θ) ln{E_w[ exp(θ v(W'))]}

"""

function T(v, model)
    (; n, w_vals, P, β, c, θ) = model
    h = c .+ (β / θ) * log.(P * (exp.(θ * v)))
    e = w_vals ./ (1 - β)
    return max.(e, h)
end

" Get a v-greedy policy."
function get_greedy(v, model)
    (; n, w_vals, P, β, c, θ) = model
    σ = w_vals / (1 - β) .>= c .+ (β / θ) * log.(P * (exp.(θ * v)))
    return σ
end

"Solve the infinite-horizon Markov job search model by VFI."
function vfi(model) 
    v_init = zero(model.w_vals)  
    v_star = successive_approx(v -> T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
end



# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16


function plot_main(; theta_vals=(-10, 0.0001, 0.1),
                     savefig=false, 
                     figname="../figures/risk_sensitive_js.pdf")

    fig, axes = plt.subplots(length(theta_vals), 1, figsize=(9, 22))

    for (θ, ax) in zip(theta_vals, axes)
        model = create_markov_js_model(θ=θ)
        (; n, w_vals, P, β, c, θ) = model
        v_star, σ_star = vfi(model)

        h_star = c .+ (β / θ) * log.(P * (exp.(θ * v_star)))
        e = w_vals / (1 - β)

        ax.plot(w_vals, h_star, lw=4, ls="--", alpha=0.4, label=L"h^*(w)")
        ax.plot(w_vals, e, lw=4, ls="--", alpha=0.4, label=L"w/(1-\beta)")
        ax.plot(w_vals, max.(e, h_star), "k-", alpha=0.7, label=L"v^*(w)")
        ax.set_title(L"\theta = " * "$θ", fontsize=fontsize)
        ax.legend(frameon=false, fontsize=fontsize)
        ax.set_xlabel(L"w", fontsize=fontsize)
    end

    fig.tight_layout()
    plt.show()

    if savefig
        fig.savefig(figname)
    end
end

