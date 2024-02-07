"""

The base model has Bellman equation

    (Tv)(w, e) = max_{0 <= s <= w} B(w, e, s, v)

where

    B(w, e, s, v) = { r(w, e, s)^α + β [Σ_e' v^γ(s, e') φ(e')]^{α/γ} }^{1/γ}


with α = 1 - 1/ψ and r(w, e, s) = (w - s + e).

We take φ to be the Binomial distribution on e_grid = (e_1, ..., e_n}) with e_1 > 0.

In particular, φ(k) = Prob{E = e_k} and φ is Bin(n-1, p)

Let α = 1 - 1 / ψ  and γ = 1 - σ 


Basu and Bundick use ψ = 0.8, β = 0.994 and σ = 30.

SSY use ψ = 1.97, β = 0.999 and σ = -7.89.


We also study the subordinate model

    (B_σ h)(w) = { Σ_e (r(w, e, σ(w))^α + β * h(σ(w))^α)^(γ/α) φ(e) }^(1/γ)

The optimal policy is found by solving the G_max step, with h as the fixed point
of B_σ and 

    σ(w, e) = argmax_s { r(w, e, s)^α + β * h(s)^α }^(1/α)


"""

using QuantEcon, Distributions, LinearAlgebra, IterTools

function create_ez_model(; ψ=1.97,  # elasticity of intertemp. substitution
                           β=0.96,  # discount factor
                           γ=-7.89, # risk aversion parameter
                           n=80,    # size of range(e)
                           p=0.5,
                           e_max=0.5,
                           w_size=50, w_max=2)
    α = 1 - 1/ψ
    θ = γ / α
    b = Binomial(n - 1, p)
    φ = [pdf(b, k) for k in 0:(n-1)]
    e_grid = LinRange(1e-5, e_max, n)
    w_grid = LinRange(0, w_max, w_size)
    return (; α, β, γ, θ, φ, e_grid, w_grid)
end

"Action-value aggregator for the original model."
function B(i, j, k, v, model)
    (; α, β, γ, θ, φ, e_grid, w_grid) = model
    w, e, s = w_grid[i], e_grid[j], w_grid[k]
    value = -Inf
    if s <= w
        Rv = @views dot(v[k, :].^γ, φ)^(1/γ)
        value = ((w - s + e)^α + β * Rv^α)^(1/α)
    end
    return value
end

"Action-value aggregator for the subordinate model."
function B(i, k, h, model)
    (; α, β, γ, θ, φ, e_grid, w_grid) = model
    w, s = w_grid[i], w_grid[k]
    G(e) = ((w - s + e)^α + β * h[k]^α)^(1/α)
    value = s <= w ? dot(G.(e_grid).^γ, φ)^(1/γ) : -Inf
    return value
end

"G maximization step to find the optimal policy of the original ADP."
function G_max(h, model)

    w_n, e_n = length(model.w_grid), length(model.e_grid)
    function G_obj(i, j, k) 
        w, e, s = w_grid[i], e_grid[j], w_grid[k]
        value = -Inf
        if s <= w
            value = ((w - s + e)^α + β * h[k]^α)^(1/α)
        end
        return value
    end
    σ_star_mod = Array{Int32}(undef, w_n, e_n)
    for i in 1:w_n
        for j in 1:e_n
            _, σ_star_mod[i, j] = findmax(G_obj(i, j, k) for k in 1:w_n)
        end
    end
    return σ_star_mod
end

