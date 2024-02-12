"""

Continuous time job search.

Since Julia uses 1-based indexing, job status is

    s = 1 for unemployed and s = 2 for employed

The policy function has the form 

    σ[j] = optimal choice in when s = 1

We use σ[j] = 1 for reject and 2 for accept (1-based indexing)

"""


using QuantEcon, Distributions, LinearAlgebra, IterTools

function create_js_model(; α=0.1,   # separation rate
                           κ=1.0,   # offer rate
                           δ=0.1,   # discount rate
                           n=100,   # wage grid size
                           ρ=0.9,   # wage persistence
                           ν=0.2,   # wage volatility
                           c=1.0)   # unemployment compensation
    mc = tauchen(n, ρ, ν)
    w_vals, P = exp.(mc.state_values), mc.p

    function Π(s, j, a, s′, j′)
        a -= 1  # set a to 0:1 (reject:accept)
        if s == 1 && s′ == 1
            out = P[j, j′] * (1 - a)
        elseif s == 1 && s′ == 2
            out = P[j, j′] * a
        elseif s == 2 && s′ == 1
            out = P[j, j′]
        else
            out = 0.0
        end
        return out
    end

    Q = Array{Float64}(undef, 2, n, 2, 2, n)
    indices = product(1:2, 1:n, 1:2, 1:2, 1:n)
    for (s, j, a, s′, j′) in indices
         λ = (s == 1) ? κ : α
         Q[s, j, a, s′, j′] = λ * 
             (Π(s, j, a, s′, j′) - (s == s′ && j == j′))
    end

    return (; n, w_vals, P, Q, δ, κ, c, α)
end

function B(s, j, a, v, model)
    (; n, w_vals, P, Q, δ, κ, c, α) = model
    r = (s == 1) ? c : w_vals[j]
    indices = product(1:2, 1:n)
    continuation_value = 0.0
    for (s′, j′) in indices
        continuation_value += v[s′, j′] * Q[s, j, a, s′, j′] 
    end
    return r + continuation_value
end


"Compute a v-greedy policy."
function get_greedy(v, model)
    (; n, w_vals, P, Q, δ, κ, c, α) = model
    σ = Array{Int8}(undef, n)
    for j in 1:n
        _, σ[j] = findmax(B(1, j, a, v, model) for a in 1:2)
    end
    return σ
end


"Approximate lifetime value of policy σ."
function get_value(σ, model)
    (; n, w_vals, P, Q, δ, κ, c, α) = model
    # Set up matrices
    A = Array{Float64}(undef, 2, n, 2, n)  # A = I δ - Q_σ
    r_σ = Array{Float64}(undef, 2, n)
    indices = product(1:2, 1:n)
    for (s, j) in indices
        r_σ[s, j] = (s == 1) ? c : w_vals[j]
    end
    indices = product(1:2, 1:n, 1:2, 1:n)
    for (s, j, s′, j′) in indices
        A[s, j, s′, j′] = δ * (s == s′ && j == j′) - Q[s, j, σ[j], s′, j′]
    end
    # Reshape for matrix algebra
    A = reshape(A, 2 * n, 2 * n)
    r_σ = reshape(r_σ, 2 * n)
    # Solve for v_σ = (I δ - Q_σ)^{-1} r_σ
    v_σ = A \ r_σ   
    # Convert to shape (2, n) and return
    v_σ = reshape(v_σ, 2, n)
    return v_σ
end

"Howard policy iteration routine."
function policy_iteration(v_init, 
                          model; 
                          tolerance=1e-9, 
                          max_iter=1_000)
    v = v_init
    error = tolerance + 1
    k = 1
    while error > tolerance && k < max_iter
        last_v = v
        σ = get_greedy(v, model)
        v = get_value(σ, model)
        error = maximum(abs.(v - last_v))
        println("Completed iteration $k with error $error.")
        k += 1
    end
    return v, get_greedy(v, model)
end


# == Figures == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=14


function plot_policy(; savefig=false, figname="cont_time_js_pol.pdf")
    model = create_js_model()
    (; n, w_vals, P, Q, δ, κ, c, α) = model

    v_init = ones(2, model.n)
    @time v_star, σ_star = policy_iteration(v_init, model)
    σ_star = σ_star .- 1  # Convert to 0:1 choices

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(w_vals, σ_star)
    ax.set_xlabel("wage offer", fontsize=fontsize)
    ax.set_yticks((0, 1))
    ax.set_ylabel("action (reject/accept)", fontsize=fontsize)
    if savefig
        plt.savefig(figname)
    end
    plt.show()
end


function plot_reswage(; savefig=false, figname="cont_time_js_res.pdf")
    α_vals = LinRange(0.05, 1.0, 100)
    res_wages_alpha = []
    for α in α_vals
        model = create_js_model(α=α)
        local (; n, w_vals, P, Q, δ, κ, c, α) = model

        v_init = ones(2, model.n)
        local v_star, σ_star = policy_iteration(v_init, model)
        σ_star = σ_star .- 1  # Convert to 0:1 choices
        
        w_idx = searchsortedfirst(σ_star, 1)
        w_bar = w_vals[w_idx]
        push!(res_wages_alpha, w_bar)

    end

    κ_vals = LinRange(0.5, 1.5, 100)
    res_wages_kappa = []
    for κ in κ_vals
        model = create_js_model(κ=κ)
        local (; n, w_vals, P, Q, δ, κ, c, α) = model

        v_init = ones(2, model.n)
        local v_star, σ_star = policy_iteration(v_init, model)
        σ_star = σ_star .- 1  # Convert to 0:1 choices
        
        w_idx = searchsortedfirst(σ_star, 1)
        w_bar = w_vals[w_idx]
        push!(res_wages_kappa, w_bar)
    end


    δ_vals = LinRange(0.05, 1.0, 100)
    res_wages_delta = []
    for δ in δ_vals
        model = create_js_model(δ=δ)
        local (; n, w_vals, P, Q, δ, κ, c, α) = model

        v_init = ones(2, model.n)
        local v_star, σ_star = policy_iteration(v_init, model)
        σ_star = σ_star .- 1  # Convert to 0:1 choices
        
        w_idx = searchsortedfirst(σ_star, 1)
        w_bar = w_vals[w_idx]
        push!(res_wages_delta, w_bar)

    end


    c_vals = LinRange(0.5, 1.5, 100)
    res_wages_c = []
    for c in c_vals
        model = create_js_model(c=c)
        local (; n, w_vals, P, Q, δ, κ, c, α) = model

        v_init = ones(2, model.n)
        local v_star, σ_star = policy_iteration(v_init, model)
        σ_star = σ_star .- 1  # Convert to 0:1 choices
        
        w_idx = searchsortedfirst(σ_star, 1)
        w_bar = w_vals[w_idx]
        push!(res_wages_c, w_bar)
    end

    multi_fig, axes = plt.subplots(2, 2, figsize=(9, 5))

    ax = axes[1, 1]
    ax.plot(α_vals, res_wages_alpha)
    ax.set_xlabel("separation rate", fontsize=fontsize)
    ax.set_ylabel("res. wage", fontsize=fontsize)

    ax = axes[1, 2]
    ax.plot(κ_vals, res_wages_kappa)
    ax.set_xlabel("offer rate", fontsize=fontsize)
    ax.set_ylabel("res. wage", fontsize=fontsize)

    ax = axes[2, 1]
    ax.plot(δ_vals, res_wages_delta)
    ax.set_xlabel("discount rate", fontsize=fontsize)
    ax.set_ylabel("res. wage", fontsize=fontsize)

    ax = axes[2, 2]
    ax.plot(c_vals, res_wages_c)
    ax.set_xlabel("unempl. compensation", fontsize=fontsize)
    ax.set_ylabel("res. wage", fontsize=fontsize)

    multi_fig.tight_layout()

    if savefig
        plt.savefig(figname)
    end
    plt.show()
end
