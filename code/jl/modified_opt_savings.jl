using QuantEcon, LinearAlgebra, IterTools

function create_savings_model(; β=0.98, γ=2.5,  
                                w_min=0.01, w_max=20.0, w_size=100,
                                ρ=0.9, ν=0.1, y_size=20,
                                η_min=0.75, η_max=1.25, η_size=2)
    η_grid = LinRange(η_min, η_max, η_size)  
    ϕ = ones(η_size) * (1 / η_size)  # Uniform distributoin
    w_grid = LinRange(w_min, w_max, w_size)  
    mc = tauchen(y_size, ρ, ν)
    y_grid, Q = exp.(mc.state_values), mc.p
    return (; β, γ, η_grid, ϕ, w_grid, y_grid, Q)
end



## == Functions for regular OPI == ##

"""
B(w, y, η, w′) = u(w + y - w′/η)) + β Σ v(w′, y′, η′) Q(y, y′) ϕ(η′)
"""
function B(i, j, k, l, v, model)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    w, y, η, w′ = w_grid[i], y_grid[j], η_grid[k], w_grid[l]
    u(c) = c^(1-γ)/(1-γ)
    c = w + y - (w′/ η)
    exp_value = 0.0
    for m in eachindex(y_grid)
        for n in eachindex(η_grid)
            exp_value += v[l, m, n] * Q[j, m] * ϕ[n]
        end
    end
    return c > 0 ? u(c) + β * exp_value : -Inf
end

"The policy operator."
function T_σ(v, σ, model)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    grids = w_grid, y_grid, η_grid
    w_idx, y_idx, η_idx = (eachindex(g) for g in grids)
    v_new = similar(v)
    for (i, j, k) in product(w_idx, y_idx, η_idx)
        v_new[i, j, k] = B(i, j, k, σ[i, j, k], v, model) 
    end
    return v_new
end

"Compute a v-greedy policy."
function get_greedy(v, model)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    w_idx, y_idx, η_idx = (eachindex(g) for g in (w_grid, y_grid, η_grid))
    σ = Array{Int32}(undef, length(w_idx), length(y_idx), length(η_idx))
    for (i, j, k) in product(w_idx, y_idx, η_idx)
        _, σ[i, j, k] = findmax(B(i, j, k, l, v, model) for l in w_idx)
    end
    return σ
end


"Optimistic policy iteration routine."
function optimistic_policy_iteration(model; tolerance=1e-5, m=100)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    v = zeros(length(w_grid), length(y_grid), length(η_grid))
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
    return get_greedy(v, model)
end


## == Functions for modified OPI == ##

"D(w, y, η, w′, g) = u(w + y - w′/η) + β g(y, w′)."
@inline function D(i, j, k, l, g, model)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    w, y, η, w′ = w_grid[i], y_grid[j], η_grid[k], w_grid[l]
    u(c) = c^(1-γ)/(1-γ)
    c = w + y - (w′/η)
    return c > 0 ? u(c) + β * g[j, l] : -Inf
end


"Compute a g-greedy policy."
function get_g_greedy(g, model)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    w_idx, y_idx, η_idx = (eachindex(g) for g in (w_grid, y_grid, η_grid))
    σ = Array{Int32}(undef, length(w_idx), length(y_idx), length(η_idx))
    for (i, j, k) in product(w_idx, y_idx, η_idx)
        _, σ[i, j, k] = findmax(D(i, j, k, l, g, model) for l in w_idx)
    end
    return σ
end


"The modified policy operator."
function R_σ(g, σ, model)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    w_idx, y_idx, η_idx = (eachindex(g) for g in (w_grid, y_grid, η_grid))
    g_new = similar(g)
    for (j, i′) in product(y_idx, w_idx)  # j indexes y, i′ indexes w′ 
        out = 0.0
        for j′ in y_idx                   # j′ indexes y′
            for k′ in η_idx               # k′ indexes η′
                out += D(i′, j′, k′, σ[i′, j′, k′], g, model) * 
                        Q[j, j′] * ϕ[k′]
            end
        end
        g_new[j, i′] = out
    end
    return g_new
end


"Modified optimistic policy iteration routine."
function mod_opi(model; tolerance=1e-5, m=100)
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    g = zeros(length(y_grid), length(w_grid))
    error = tolerance + 1
    while error > tolerance
        last_g = g
        σ = get_g_greedy(g, model)
        for i in 1:m
            g = R_σ(g, σ, model)
        end
        error = maximum(abs.(g - last_g))
        println("OPI current error = $error")
    end
    return get_g_greedy(g, model)
end


# == Simulations and inequality measures == #

function simulate_wealth(m)

    model = create_savings_model()
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    σ_star = mod_opi(model)

    # Simulate labor income
    mc = MarkovChain(Q)
    y_idx_series = simulate(mc, m)

    # IID Markov chain with uniform draws
    l = length(η_grid)
    mc = MarkovChain(ones(l, l) * (1/l))
    η_idx_series = simulate(mc, m)

    w_idx_series = similar(y_idx_series)
    w_idx_series[1] = 1
    for t in 1:(m-1)
        i, j, k = w_idx_series[t], y_idx_series[t], η_idx_series[t]
        w_idx_series[t+1] = σ_star[i, j, k]
    end

    w_series = w_grid[w_idx_series]
    return w_series
end


function lorenz(v)  # assumed sorted vector
    S = cumsum(v)  # cumulative sums: [v[1], v[1] + v[2], ... ]
    F = (1:length(v)) / length(v)
    L = S ./ S[end]
    return (; F, L) # returns named tuple
end


gini(v) = (2 * sum(i * y for (i,y) in enumerate(v))/sum(v)
           - (length(v) + 1))/length(v)


# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16



function plot_contours(; savefig=false, 
                         figname="./figures/modified_opt_savings_1.pdf")

    model = create_savings_model()
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    σ_star = optimistic_policy_iteration(model)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    y_idx, η_idx = eachindex(y_grid), eachindex(η_grid)
    H = zeros(length(y_grid), length(η_grid))

    w_indices = (1, length(w_grid))
    titles = "low wealth", "high wealth"
    for (ax, w_idx, title) in zip(axes, w_indices, titles)

        for (i_y, i_ϵ) in product(y_idx, η_idx)
            w, y, η = w_grid[w_idx], y_grid[i_y], η_grid[i_ϵ]
            H[i_y, i_ϵ] = w_grid[σ_star[w_idx, i_y, i_ϵ]] / (w+y)
        end

        cs1 = ax.contourf(y_grid, η_grid, transpose(H), alpha=0.5)
        plt.colorbar(cs1, ax=ax) #, format="%.6f")

        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(L"y", fontsize=fontsize)
        ax.set_ylabel(L"\varepsilon", fontsize=fontsize)
    end

    plt.tight_layout()
    if savefig
        fig.savefig(figname)
    end
    plt.show()
end


function plot_policies(; savefig=false, 
                         figname="./figures/modified_opt_savings_2.pdf")

    model = create_savings_model()
    (; β, γ, η_grid, ϕ, w_grid, y_grid, Q) = model
    σ_star = mod_opi(model)
    y_bar = floor(Int, length(y_grid) / 2)  # Index of mid-point of y_grid

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, w_grid, "k--", label=L"45")

    for (i, η) in enumerate(η_grid)
        label = L"\sigma^*" * " at " * L"\eta = " * "$η"
        ax.plot(w_grid, w_grid[σ_star[:, y_bar, i]], label=label)
    end
    ax.legend(fontsize=fontsize)
    plt.show()

    plt.tight_layout()
    if savefig
        fig.savefig(figname)
    end
    plt.show()
end


function plot_time_series(; m=2_000,
                           savefig=false, 
                           figname="./figures/modified_opt_savings_ts.pdf")

    w_series = simulate_wealth(m)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_series, label=L"w_t")
    ax.legend(fontsize=fontsize)
    ax.set_xlabel("time", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end

end

function plot_histogram(; m=1_000_000,
                           savefig=false, 
                           figname="./figures/modified_opt_savings_hist.pdf")

    w_series = simulate_wealth(m)
    g = round(gini(sort(w_series)), digits=2)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(w_series, bins=40, density=true)
    ax.set_xlabel("wealth", fontsize=fontsize)
    ax.text(15, 0.7, "Gini = $g", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end

end


function plot_lorenz(; m=1_000_000,
                           savefig=false, 
                           figname="./figures/modified_opt_savings_lorenz.pdf")

    w_series = simulate_wealth(m)
    (; F, L) = lorenz(sort(w_series))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(F, F, label="Lorenz curve, equality")
    ax.plot(F, L, label="Lorenz curve, wealth distribution")
    ax.legend()
    plt.show()
    if savefig
        fig.savefig(figname)
    end

end
