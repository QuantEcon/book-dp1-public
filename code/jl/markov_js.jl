"""
Infinite-horizon job search with Markov wage draws.

"""

using QuantEcon, LinearAlgebra
include("s_approx.jl")

"Creates an instance of the job search model with Markov wages."
function create_markov_js_model(;
        n=200,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.98,      # discount factor
        c=1.0        # unemployment compensation
    )
    mc = tauchen(n, ρ, ν)
    w_vals, P = exp.(mc.state_values), mc.p
    return (; n, w_vals, P, β, c)
end

" The Bellman operator Tv = max{e, c + β P v} with e(w) = w / (1-β)."
function T(v, model)
    (; n, w_vals, P, β, c) = model
    h = c .+ β * P * v
    e = w_vals ./ (1 - β)
    return max.(e, h)
end

" Get a v-greedy policy."
function get_greedy(v, model)
    (; n, w_vals, P, β, c) = model
    σ = w_vals / (1 - β) .>= c .+ β * P * v
    return σ
end

"Solve the infinite-horizon Markov job search model by VFI."
function vfi(model) 
    v_init = zero(model.w_vals)  
    v_star = successive_approx(v -> T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
end

        

# == Policy iteration == #

"Get the value of policy σ."
function get_value(σ, model)
    (; n, w_vals, P, β, c) = model
    e = w_vals ./ (1 - β)
    K_σ = β .* (1 .- σ) .* P
    r_σ = σ .* e .+ (1 .- σ) .* c
    return (I - K_σ) \ r_σ
end


    
"Howard policy iteration routine."
function policy_iteration(model)
    σ = Vector{Bool}(undef, model.n)
    i, error = 0, 1.0
    while error > 0
        v_σ = get_value(σ, model)
        σ_new = get_greedy(v_σ, model)
        error = maximum(abs.(σ_new - σ))
        σ = σ_new
        i = i + 1
        println("Concluded loop $i with error $error.")
    end
    return σ
end


# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

default_model = create_markov_js_model()


function plot_main(; model=default_model,
                     method="vfi", 
                     savefig=false, 
                     figname="../figures/markov_js_1.pdf")
    (; n, w_vals, P, β, c) = model


    if method == "vfi"
        v_star, σ_star = vfi(model)
    else
        σ_star = policy_iteration(model)
        v_star = get_value(σ_star, model)
    end

    h_star = c .+ β * P * v_star
    e = w_vals / (1 - β)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_vals, h_star, lw=4, ls="--", alpha=0.4, label=L"h^*(w)")
    ax.plot(w_vals, e, lw=4, ls="--", alpha=0.4, label=L"w/(1-\beta)")
    ax.plot(w_vals, max.(e, h_star), "k-", alpha=0.7, label=L"v^*(w)")
    ax.legend(frameon=false, fontsize=fontsize)
    ax.set_xlabel(L"w", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

