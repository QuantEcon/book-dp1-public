"""
Job search with Markov wage draws and quantile preferences.

"""

using QuantEcon
include("quantile_function.jl")

"Creates an instance of the job search model."
function create_markov_js_model(;
        n=100,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.98,      # discount factor
        c=1.0,       # unemployment compensation
        τ=0.5        # quantile parameter
    )
    mc = tauchen(n, ρ, ν)
    w_vals, P = exp.(mc.state_values), mc.p
    return (; n, w_vals, P, β, c, τ)
end

"""
The policy operator 

    (T_σ v)(w) = σ(w) (w / (1-β)) + (1 - σ(w))(c + β (R_τ v)(w))

"""
function T_σ(v, σ, model)
    (; n, w_vals, P, β, c, τ) = model
    h = c .+ β * R(τ, v, P)
    e = w_vals ./ (1 - β)
    return σ .* e + (1 .- σ) .* h
end

" Get a v-greedy policy."
function get_greedy(v, model)
    (; n, w_vals, P, β, c, τ) = model
    σ = w_vals / (1 - β) .≥ c .+ β * R(τ, v, P)
    return σ
end


"Optimistic policy iteration routine."
function optimistic_policy_iteration(model; tolerance=1e-5, m=100)
    (; n, w_vals, P, β, c, τ) = model
    v = ones(n)
    error = tolerance + 1
    while error > tolerance
        last_v = v
        σ = get_greedy(v, model)
        for i in 1:m
            v = T_σ(v, σ, model)
        end
        error = maximum(abs.(v - last_v))
        println("OPI current error = $error")
    end
    return v, get_greedy(v, model)
end



# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16


function plot_main(; tau_vals=(0.1, 0.25, 0.5, 0.6, 0.7, 0.8),
                     savefig=false, 
                     figname="../figures/quantile_js.pdf")

    w_star_vals = zeros(length(tau_vals))

    for (τ_idx, τ) in enumerate(tau_vals)
        model = create_markov_js_model(τ=τ)
        (; n, w_vals, P, β, c, τ) = model
        v_star, σ_star = optimistic_policy_iteration(model)
        for i in 1:n
            if σ_star[i] > 0
                w_star_vals[τ_idx] = w_vals[i]
                break
            end
        end
    end

    model = create_markov_js_model()
    (; n, w_vals, P, β, c, τ) = model
    mc = MarkovChain(model.P)
    s = stationary_distributions(mc)[1]

    fig, ax = plt.subplots()
    ax.plot(tau_vals, w_star_vals, "k--", lw=2, alpha=0.7, label="reservation wage")
    ax.barh(w_vals, 32 * s, alpha=0.05, align="center")
    ax.legend(frameon=false, fontsize=fontsize, loc="upper center")
    ax.set_xlabel("quantile", fontsize=fontsize)
    ax.set_ylabel("wages", fontsize=fontsize)

    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

