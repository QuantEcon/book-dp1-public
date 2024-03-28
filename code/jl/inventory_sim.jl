using Distributions, QuantEcon, IterTools

function create_inventory_model(; S=100,  # Order size
                                  s=10,   # Order threshold
                                  p=0.4)  # Demand parameter
    ϕ = Geometric(p)
    h(x, d) = max(x - d, 0) + S * (x <= s)
    return (; S, s, ϕ, h)
end

"Simulate the inventory process."
function sim_inventories(model; ts_length=200)
    (; S, s, ϕ, h) = model
    X = Vector{Int32}(undef, ts_length)
    X[1] = S  # Initial condition
    for t in 1:(ts_length-1)
        X[t+1] = h(X[t], rand(ϕ))
    end
    return X
end

"Compute the transition probabilities and state."
function compute_mc(model; d_max=100)
    (; S, s, ϕ, h) = model
    n = S + s + 1  # Size of state space
    state_vals = collect(0:(S + s))
    P = Matrix{Float64}(undef, n, n)
    for (i, j) in product(1:n, 1:n)
        P[i, j] = sum((h(i-1, d) == j-1) * pdf(ϕ, d) for d in 0:d_max)
    end
    return MarkovChain(P, state_vals)
end

"Compute the stationary distribution of the model."
function compute_stationary_dist(model)
    mc = compute_mc(model)
    return mc.state_values, stationary_distributions(mc)[1]
end


# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering


function plot_ts(model; fontsize=16, 
                   figname="../figures/inventory_sim_1.pdf",
                   savefig=false)
    (; S, s, ϕ, h) = model
    X = sim_inventories(model)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(X, label=L"X_t", lw=3, alpha=0.6)
    ax.set_xlabel(L"t", fontsize=fontsize)
    ax.set_ylabel("inventory", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_ylim(0, S + s + 20)

    plt.show()
    if savefig == true
        fig.savefig(figname)
    end
end


function plot_hist(model; fontsize=16, 
                   figname="../figures/inventory_sim_2.pdf",
                   savefig=false)
    (; S, s, ϕ, h) = model
    state_values, ψ_star = compute_stationary_dist(model) 
    X = sim_inventories(model; ts_length=1_000_000)
    histogram = [mean(X .== i) for i in state_values]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(state_values, ψ_star, "k-",  lw=3, alpha=0.7, label=L"\psi^*")
    ax.bar(state_values, histogram, alpha=0.7, label="frequency")
    ax.set_xlabel("state", fontsize=fontsize)

    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_ylim(0, 0.015)

    plt.show()
    if savefig == true
        fig.savefig(figname)
    end
end

