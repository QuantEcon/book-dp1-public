---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.9
---

(Chapter 7: Nonlinear Valuation)=
```{raw} jupyter
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```
# Chapter 7: Nonlinear Valuation


```{contents} Contents
:depth: 2
```


```{code-cell} julia
:tags: ["remove-cell"]
using Pkg;
Pkg.activate("./");

using PyCall;
pygui(:tk);
```

## rs_utility.jl
```{code-cell} julia
include("s_approx.jl")
using LinearAlgebra, QuantEcon

function create_rs_utility_model(;
        n=180,      # size of state space
        β=0.95,     # time discount factor
        ρ=0.96,     # correlation coef in AR(1)
        σ=0.1,      # volatility
        θ=-1.0)     # risk aversion
    mc = tauchen(n, ρ, σ, 0, 10)  # n_std = 10
    x_vals, P = mc.state_values, mc.p 
    r = x_vals      # special case u(c(x)) = x
    return (; β, θ, ρ, σ, r, x_vals, P)
end

function K(v, model)
    (; β, θ, ρ, σ, r, x_vals, P) = model
    return r + (β/θ) * log.(P * (exp.(θ*v)))
end

function compute_rs_utility(model)
    (; β, θ, ρ, σ, r, x_vals, P) = model
    v_init = zeros(length(x_vals))
    v_star = successive_approx(v -> K(v, model), 
                               v_init, tolerance=1e-10)
    return v_star
end


# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

function plot_v(; savefig=false, 
                  figname="./figures/rs_utility_1.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    model = create_rs_utility_model()
    (; β, θ, ρ, σ, r, x_vals, P) = model

    a = 1/(1 - (ρ*β))
    b = (β /(1 - β)) * (θ/2) * (a*σ)^2 
    
    v_star = compute_rs_utility(model)
    v_star_a = a * x_vals .+ b
    ax.plot(x_vals, v_star, lw=2, alpha=0.7, label="approximate fixed point")
    ax.plot(x_vals, v_star_a, "k--", lw=2, alpha=0.7, label=L"v(x)=ax + b")
    ax.set_xlabel(L"x", fontsize=fontsize)

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    if savefig
        fig.savefig(figname)
    end
end




function plot_multiple_v(; savefig=false, 
                  figname="./figures/rs_utility_2.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    σ_vals = 0.05, 0.1

    for σ in σ_vals
        model = create_rs_utility_model(σ=σ)
        (; β, θ, r, x_vals, P) = model
        v_star = compute_rs_utility(model)
        ax.plot(x_vals, v_star, lw=2, alpha=0.7, label=L"\sigma="*"$σ")
        ax.set_xlabel(L"x", fontsize=fontsize)
        ax.set_ylabel(L"v(x)", fontsize=fontsize)
    end

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    if savefig
        fig.savefig(figname)
    end
end


```

```{code-cell} julia
plot_v(savefig=true)
```

```{code-cell} julia
plot_multiple_v(savefig=true)
```
## ez_utility.jl
```{code-cell} julia
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
    if savefig
        fig.savefig(figname)
    end
end


```

```{code-cell} julia
plot_convergence(savefig=true)
```

```{code-cell} julia
plot_v(savefig=true)
```

```{code-cell} julia
vary_gamma(savefig=true)
```

```{code-cell} julia
vary_alpha(savefig=true)
```
