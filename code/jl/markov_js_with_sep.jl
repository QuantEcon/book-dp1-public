"""
Infinite-horizon job search with Markov wage draws and separation.

"""

include("s_approx.jl")
using QuantEcon, LinearAlgebra

"Creates an instance of the job search model with separation."
function create_js_with_sep_model(;
        n=200,          # wage grid size
        ρ=0.9, ν=0.2,   # wage persistence and volatility
        β=0.98, α=0.1,  # discount factor and separation rate
        c=1.0)          # unemployment compensation
    mc = tauchen(n, ρ, ν)
    w_vals, P = exp.(mc.state_values), mc.p
    return (; n, w_vals, P, β, c, α)
end

" The Bellman operator for the value of being unemployed."
function T(v, model)
    (; n, w_vals, P, β, c, α) = model
    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * P * v)
    reject = c .+ β * P * v
    return max.(accept, reject)
end

" Get a v-greedy policy."
function get_greedy(v, model)
    (; n, w_vals, P, β, c, α) = model
    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * P * v)
    reject = c .+ β * P * v
    σ = accept .>= reject
    return σ
end

"Solve by VFI."
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

default_model = create_js_with_sep_model()


function plot_main(; model=default_model,
                     method="vfi", 
                     savefig=false, 
                     figname="../figures/markov_js_with_sep_1.pdf")
    (; n, w_vals, P, β, c, α) = model
    v_star, σ_star = vfi(model)

    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * P * v_star)
    h_star = c .+ β * P * v_star

    w_star = Inf
    for (i, w) in enumerate(w_vals)
        if accept[i] ≥ h_star[i]
            w_star = w
            break
        end
    end
    @assert w_star < Inf "Agent never accepts"

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_vals, h_star, lw=4, ls="--", alpha=0.4, label="continuation value")
    ax.plot(w_vals, accept, lw=4, ls="--", alpha=0.4, label="stopping value")
    ax.plot(w_vals, v_star, "k-", alpha=0.7, label=L"v_u^*(w)")
    ax.legend(frameon=false, fontsize=fontsize)
    ax.set_xlabel(L"w", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

function plot_w_stars(; α_vals=LinRange(0.0, 1.0, 10),
                        savefig=false, 
                        figname="../figures/markov_js_with_sep_2.pdf")

    w_star_vec = similar(α_vals)
    for (i_α, α) in enumerate(α_vals)
        print(i_α, α)
        model = create_js_with_sep_model(α=α)
        (; n, w_vals, P, β, c, α) = model
        v_star, σ_star = vfi(model)

        d = 1 / (1 - β * (1 - α))
        accept = d * (w_vals + α * β * P * v_star)
        h_star = c .+ β * P * v_star

        w_star = Inf
        for (i_w, w) in enumerate(w_vals)
            if accept[i_w] ≥ h_star[i_w]
                w_star = w
                break
            end
        end
        @assert w_star < Inf "Agent never accepts"
        w_star_vec[i_α] = w_star
    end

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(α_vals, w_star_vec, lw=2, alpha=0.6, label="reservation wage")
    ax.legend(frameon=false, fontsize=fontsize)
    ax.set_xlabel(L"\alpha", fontsize=fontsize)
    ax.set_xlabel(L"w", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


