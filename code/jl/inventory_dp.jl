include("s_approx.jl")
using Distributions

f(x, a, d) = max(x - d, 0) + a  # Inventory update

function create_inventory_model(; β=0.98,     # discount factor
                                  K=40,       # maximum inventory
                                  c=0.2, κ=2, # cost paramters
                                  p=0.6)      # demand parameter
    ϕ(d) = (1 - p)^d * p        # demand distribution
    x_vals = collect(0:K)       # set of inventory levels
    return (; β, K, c, κ, p, ϕ, x_vals)
end

"The function B(x, a, v) = r(x, a) + β Σ_x′ v(x′) P(x, a, x′)."
function B(x, a, v, model; d_max=100)
    (; β, K, c, κ, p, ϕ, x_vals) = model
    revenue = sum(min(x, d) * ϕ(d) for d in 0:d_max) 
    current_profit = revenue - c * a - κ * (a > 0)
    next_value = sum(v[f(x, a, d) + 1] * ϕ(d) for d in 0:d_max)
    return current_profit + β * next_value
end

"The Bellman operator."
function T(v, model)
    (; β, K, c, κ, p, ϕ, x_vals) = model
    new_v = similar(v)
    for (x_idx, x) in enumerate(x_vals)
        Γx = 0:(K - x) 
        new_v[x_idx], _ = findmax(B(x, a, v, model) for a in Γx)
    end
    return new_v
end

"Get a v-greedy policy."
function get_greedy(v, model)
    (; β, K, c, κ, p, ϕ, x_vals) = model
    σ_star = zero(x_vals)
    for (x_idx, x) in enumerate(x_vals)
        Γx = 0:(K - x) 
        _, a_idx = findmax(B(x, a, v, model) for a in Γx)
        σ_star[x_idx] = Γx[a_idx]
    end
    return σ_star
end

"Use successive_approx to get v_star and then compute greedy."
function solve_inventory_model(v_init, model)
    (; β, K, c, κ, p, ϕ, x_vals) = model
    v_star = successive_approx(v -> T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
end

# == Plots == # 

using PyPlot
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

# Create an instance of the model and solve it
model = create_inventory_model()
(; β, K, c, κ, p, ϕ, x_vals) = model
v_init = zeros(length(x_vals))
v_star, σ_star = solve_inventory_model(v_init, model)

"Simulate given the optimal policy."
function sim_inventories(ts_length=400, X_init=0)
    G = Geometric(p)
    X = zeros(Int32, ts_length)
    X[1] = X_init
    for t in 1:(ts_length-1)
        D = rand(G)
        X[t+1] = f(X[t],  σ_star[X[t] + 1], D)
    end
    return X
end


function plot_vstar_and_opt_policy(; fontsize=16, 
                   figname="../figures/inventory_dp_vs.pdf",
                   savefig=false)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))

    ax = axes[1]
    ax.plot(0:K, v_star, label=L"v^*")
    ax.set_ylabel("value", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)

    ax = axes[2]
    ax.plot(0:K, σ_star, label=L"\sigma^*")
    ax.set_xlabel("inventory", fontsize=fontsize)
    ax.set_ylabel("optimal choice", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)
    plt.show()
    if savefig == true
        fig.savefig(figname)
    end
end

function plot_ts(; fontsize=16, 
                   figname="../figures/inventory_dp_ts.pdf",
                   savefig=false)
    X = sim_inventories()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(X, label=L"X_t", alpha=0.7)
    ax.set_xlabel(L"t", fontsize=fontsize)
    ax.set_ylabel("inventory", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_ylim(0, maximum(X)+4)
    plt.show()
    if savefig == true
        fig.savefig(figname)
    end
end

