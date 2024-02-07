"""
Inventory management model with state-dependent discounting.  
The discount factor takes the form β_t = Z_t, where (Z_t) is 
a discretization of the Gaussian AR(1) process 

    X_t = ρ X_{t-1} + b + ν W_t.

"""

include("s_approx.jl")
using LinearAlgebra, Distributions, QuantEcon

function create_sdd_inventory_model(; 
            ρ=0.98, ν=0.002, n_z=20, b=0.97,  # Z state parameters
            K=40, c=0.2, κ=0.8, p=0.6)        # firm and demand parameters
    ϕ(d) = (1 - p)^d * p                      # demand pdf
    y_vals = collect(0:K)                     # inventory levels
    mc = tauchen(n_z, ρ, ν)
    z_vals, Q = mc.state_values .+ b, mc.p
    ρL = maximum(abs.(eigvals(z_vals .* Q)))     
    @assert  ρL < 1 "Error: ρ(L) ≥ 1."    # check ρ(L) < 1
    return (; K, c, κ, p, ϕ, y_vals, z_vals, Q)
end

m(y) = max(y, 0)  # Convenience function

"The function B(x, a, v) = r(x, a) + β(x) Σ_x′ v(x′) P(x, a, x′)."
function B(x, i_z, a, v, model; d_max=100)
    (; K, c, κ, p, ϕ, y_vals, z_vals, Q) = model
    z = z_vals[i_z]
    revenue = sum(min(x, d)*ϕ(d) for d in 0:d_max)
    current_profit =  revenue - c * a - κ * (a > 0)
    cv = 0.0
    for i_z′ in eachindex(z_vals)
        for d in 0:d_max
            cv += v[m(x - d) + a + 1, i_z′] * ϕ(d) * Q[i_z, i_z′]
        end
    end
    return current_profit + z * cv
end

"The Bellman operator."
function T(v, model)
    (; K, c, κ, p, ϕ, y_vals, z_vals, Q) = model
    new_v = similar(v)
    for (i_z, z) in enumerate(z_vals)
        for (i_y, y) in enumerate(y_vals)
            Γy = 0:(K - y) 
            new_v[i_y, i_z], _ = findmax(B(y, i_z, a, v, model) for a in Γy)
        end
    end
    return new_v
end

"Get a v-greedy policy.  Returns a zero-based array."
function get_greedy(v, model)
    (; K, c, κ, p, ϕ, y_vals, z_vals, Q) = model
    n_z = length(z_vals)
    σ_star = zeros(Int32, K+1, n_z)
    for (i_z, z) in enumerate(z_vals)
        for (i_y, y) in enumerate(y_vals)
            Γy = 0:(K - y) 
            _, i_a = findmax(B(y, i_z, a, v, model) for a in Γy)
            σ_star[i_y, i_z] = Γy[i_a]
        end
    end
    return σ_star
end


"Use successive_approx to get v_star and then compute greedy."
function solve_inventory_model(v_init, model)
    (; K, c, κ, p, ϕ, y_vals, z_vals, Q) = model
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
model = create_sdd_inventory_model()
(; K, c, κ, p, ϕ, y_vals, z_vals, Q) = model
n_z = length(z_vals)
v_init = zeros(Float64, K+1, n_z)
println("Solving model.")
v_star, σ_star = solve_inventory_model(v_init, model)
z_mc = MarkovChain(Q, z_vals)

"Simulate given the optimal policy."
function sim_inventories(ts_length; X_init=0)
    i_z = simulate_indices(z_mc, ts_length, init=1)
    G = Geometric(p)
    X = zeros(Int32, ts_length)
    X[1] = X_init
    for t in 1:(ts_length-1)
        D′ = rand(G)
        X[t+1] = m(X[t] - D′) + σ_star[X[t] + 1, i_z[t]]
    end
    return X, z_vals[i_z]
end

function plot_ts(; ts_length=400,
                   fontsize=16, 
                   figname="./figures/inventory_sdd_ts.pdf",
                   savefig=false)
    X, Z = sim_inventories(ts_length)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))

    ax = axes[1]
    ax.plot(X, label="inventory", alpha=0.7)
    ax.set_xlabel(L"t", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_ylim(0, maximum(X)+3)

    # calculate interest rate from discount factors
    r = (1 ./ Z) .- 1

    ax = axes[2]
    ax.plot(r, label=L"r_t", alpha=0.7)
    ax.set_xlabel(L"t", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)
    #ax.set_ylim(0, maximum(X)+8)

    plt.tight_layout()
    plt.show()
    if savefig == true
        fig.savefig(figname)
    end
end

