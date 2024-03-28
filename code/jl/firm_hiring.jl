using QuantEcon, LinearAlgebra, IterTools

function create_hiring_model(; 
        r=0.04,                              # Interest rate
        κ=1.0,                               # Adjustment cost 
        α=0.4,                               # Production parameter
        p=1.0, w=1.0,                        # Price and wage
        l_min=0.0, l_max=30.0, l_size=100,   # Grid for labor
        ρ=0.9, ν=0.4, b=1.0,                 # AR(1) parameters
        z_size=100)                          # Grid size for shock
    β = 1/(1+r) 
    l_grid = LinRange(l_min, l_max, l_size)  
    mc = tauchen(z_size, ρ, ν, b, 6)
    z_grid, Q = mc.state_values, mc.p
    return (; β, κ, α, p, w, l_grid, z_grid, Q)
end

"""
The aggregator B is given by 

    B(l, z, l′) = r(l, z, l′) + β Σ_z′ v(l′, z′) Q(z, z′)."

where 

    r(l, z, l′) := p * z * f(l) - w * l - κ 1{l != l′}

"""
function B(i, j, k, v, model)
    (; β, κ, α, p, w, l_grid, z_grid, Q) = model
    l, z, l′ = l_grid[i], z_grid[j], l_grid[k]
    r = p * z * l^α - w * l - κ * (l != l′)
    return @views r + β * dot(v[k, :], Q[j, :]) 
end


"The policy operator."
function T_σ(v, σ, model)
    l_idx, z_idx = (eachindex(g) for g in (model.l_grid, model.z_grid))
    v_new = similar(v)
    for (i, j) in product(l_idx, z_idx)
        v_new[i, j] = B(i, j, σ[i, j], v, model) 
    end
    return v_new
end

"Compute a v-greedy policy."
function get_greedy(v, model)
    (; β, κ, α, p, w, l_grid, z_grid, Q) = model
    l_idx, z_idx = (eachindex(g) for g in (model.l_grid, model.z_grid))
    σ = Matrix{Int32}(undef, length(l_idx), length(z_idx))
    for (i, j) in product(l_idx, z_idx)
        _, σ[i, j] = findmax(B(i, j, k, v, model) for k in l_idx)
    end
    return σ
end

"Optimistic policy iteration routine."
function optimistic_policy_iteration(model; tolerance=1e-5, m=100)
    v = zeros(length(model.l_grid), length(model.z_grid))
    error = tolerance + 1
    while error > tolerance
        last_v = v
        σ = get_greedy(v, model)
        for i in 1:m
            v = T_σ(v, σ, model)
        end
        error = maximum(abs.(v - last_v))
    end
    return get_greedy(v, model)
end


# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=14

function plot_policy(; savefig=false, 
                    figname="../figures/firm_hiring_pol.pdf")
    model = create_hiring_model()
    (; β, κ, α, p, w, l_grid, z_grid, Q) = model
    σ_star = optimistic_policy_iteration(model)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(l_grid, l_grid, "k--", label=L"45")
    ax.plot(l_grid, l_grid[σ_star[:, 1]], label=L"\sigma^*(\cdot, z_1)")
    ax.plot(l_grid, l_grid[σ_star[:, end]], label=L"\sigma^*(\cdot, z_N)")
    ax.legend(fontsize=fontsize)
    plt.show()
end


function sim_dynamics(model, ts_length)

    (; β, κ, α, p, w, l_grid, z_grid, Q) = model
    σ_star = optimistic_policy_iteration(model)
    mc = MarkovChain(Q, z_grid)
    z_sim_idx = simulate_indices(mc, ts_length)
    z_sim = z_grid[z_sim_idx]
    l_sim_idx = Vector{Int32}(undef, ts_length)
    l_sim_idx[1] = 32
    for t in 1:(ts_length-1)
        l_sim_idx[t+1] = σ_star[l_sim_idx[t], z_sim_idx[t]]
    end
    l_sim = l_grid[l_sim_idx]

    y_sim = similar(l_sim)
    for (i, l) in enumerate(l_sim)
        y_sim[i] = p * z_sim[i] * l_sim[i]^α
    end

    t = ts_length - 1
    l_g, y_g, z_g = zeros(t), zeros(t), zeros(t)

    for i in 1:t
        l_g[i] = (l_sim[i+1] - l_sim[i]) / l_sim[i]
        y_g[i] = (y_sim[i+1] - y_sim[i]) / y_sim[i]
        z_g[i] = (z_sim[i+1] - z_sim[i]) / z_sim[i]
    end

    return l_sim, y_sim, z_sim, l_g, y_g, z_g

end



function plot_sim(; savefig=false, 
                    figname="../figures/firm_hiring_ts.pdf",
                    ts_length = 250)

    model = create_hiring_model()
    (; β, κ, α, p, w, l_grid, z_grid, Q) = model
    l_sim, y_sim, z_sim, l_g, y_g, z_g = sim_dynamics(model, ts_length)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(1:ts_length, l_sim, label=L"\ell_t")
    ax.plot(1:ts_length, z_sim, alpha=0.6, label=L"Z_t")
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_ylabel("employment", fontsize=fontsize)
    ax.set_xlabel("time", fontsize=fontsize)

    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


function plot_growth(; savefig=false, 
                    figname="../figures/firm_hiring_g.pdf",
                    ts_length = 10_000_000)

    model = create_hiring_model()
    (; β, κ, α, p, w, l_grid, z_grid, Q) = model
    l_sim, y_sim, z_sim, l_g, y_g, z_g = sim_dynamics(model, ts_length)

    fig, ax = plt.subplots()
    ax.hist(l_g, alpha=0.6, bins=100)
    ax.set_xlabel("growth", fontsize=fontsize)

    #fig, axes = plt.subplots(2, 1)
    #series = y_g, z_g
    #for (ax, g) in zip(axes, series)
    #    ax.hist(g, alpha=0.6, bins=100)
    #    ax.set_xlabel("growth", fontsize=fontsize)
    #end

    plt.tight_layout()
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


