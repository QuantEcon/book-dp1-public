using QuantEcon, LinearAlgebra, IterTools
include("s_approx.jl")

function create_investment_model(; 
        r=0.04,                              # Interest rate
        a_0=10.0, a_1=1.0,                   # Demand parameters
        γ=25.0, c=1.0,                       # Adjustment and unit cost 
        y_min=0.0, y_max=20.0, y_size=100,   # Grid for output
        ρ=0.9, ν=1.0,                        # AR(1) parameters
        z_size=25)                           # Grid size for shock
    β = 1/(1+r) 
    y_grid = LinRange(y_min, y_max, y_size)  
    mc = tauchen(y_size, ρ, ν)
    z_grid, Q = mc.state_values, mc.p
    return (; β, a_0, a_1, γ, c, y_grid, z_grid, Q)
end

"""
The aggregator B is given by 

    B(y, z, y′) = r(y, z, y′) + β Σ_z′ v(y′, z′) Q(z, z′)."

where 

    r(y, z, y′) := (a_0 - a_1 * y + z - c) y - γ * (y′ - y)^2

"""
function B(i, j, k, v, model)
    (; β, a_0, a_1, γ, c, y_grid, z_grid, Q) = model
    y, z, y′ = y_grid[i], z_grid[j], y_grid[k]
    r = (a_0 - a_1 * y + z - c) * y - γ * (y′ - y)^2
    return @views r + β * dot(v[k, :], Q[j, :]) 
end

"The policy operator."
function T_σ(v, σ, model)
    y_idx, z_idx = (eachindex(g) for g in (model.y_grid, model.z_grid))
    v_new = similar(v)
    for (i, j) in product(y_idx, z_idx)
        v_new[i, j] = B(i, j, σ[i, j], v, model) 
    end
    return v_new
end

"The Bellman operator."
function T(v, model)
    y_idx, z_idx = (eachindex(g) for g in (model.y_grid, model.z_grid))
    v_new = similar(v)
    for (i, j) in product(y_idx, z_idx)
        v_new[i, j] = maximum(B(i, j, k, v, model) for k in y_idx)
    end
    return v_new
end

"Compute a v-greedy policy."
function get_greedy(v, model)
    y_idx, z_idx = (eachindex(g) for g in (model.y_grid, model.z_grid))
    σ = Matrix{Int32}(undef, length(y_idx), length(z_idx))
    for (i, j) in product(y_idx, z_idx)
        _, σ[i, j] = findmax(B(i, j, k, v, model) for k in y_idx)
    end
    return σ
end

"Value function iteration routine."
function value_iteration(model; tol=1e-5)
    vz = zeros(length(model.y_grid), length(model.z_grid))
    v_star = successive_approx(v -> T(v, model), vz, tolerance=tol)
    return get_greedy(v_star, model)
end



"Get the value v_σ of policy σ."
function get_value(σ, model)
    # Unpack and set up
    (; β, a_0, a_1, γ, c, y_grid, z_grid, Q) = model
    yn, zn = length(y_grid), length(z_grid)
    n = yn * zn
    # Function to extract (i, j) from m = i + (j-1)*yn"
    single_to_multi(m) = (m-1)%yn + 1, div(m-1, yn) + 1
    # Allocate and create single index versions of P_σ and r_σ
    P_σ = zeros(n, n)
    r_σ = zeros(n)
    for m in 1:n
        i, j = single_to_multi(m)
        y, z, y′ = y_grid[i], z_grid[j], y_grid[σ[i, j]]
        r_σ[m] = (a_0 - a_1 * y + z - c) * y - γ * (y′ - y)^2
        for m′ in 1:n
            i′, j′ = single_to_multi(m′)
            if i′ == σ[i, j]
                P_σ[m, m′] = Q[j, j′]
            end
        end
    end
    # Solve for the value of σ 
    v_σ = (I - β * P_σ) \ r_σ
    # Return as multi-index array
    return reshape(v_σ, yn, zn)
end


"Howard policy iteration routine."
function policy_iteration(model)
    yn, zn = length(model.y_grid), length(model.z_grid)
    σ = ones(Int32, yn, zn)
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

"Optimistic policy iteration routine."
function optimistic_policy_iteration(model; tol=1e-5, m=100)
    v = zeros(length(model.y_grid), length(model.z_grid))
    error = tol + 1
    while error > tol
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
fontsize=12

function plot_policy()
    model = create_investment_model()
    (; β, a_0, a_1, γ, c, y_grid, z_grid, Q) = model
    σ_star = optimistic_policy_iteration(model)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(y_grid, y_grid, "k--", label=L"45")
    ax.plot(y_grid, y_grid[σ_star[:, 1]], label=L"\sigma^*(\cdot, z_1)")
    ax.plot(y_grid, y_grid[σ_star[:, end]], label=L"\sigma^*(\cdot, z_N)")
    ax.legend(fontsize=fontsize)
    plt.show()
end

function plot_sim(; savefig=false, figname="./figures/finite_lq_1.pdf")
    ts_length = 200

    fig, axes = plt.subplots(4, 1, figsize=(9, 11.2))

    for (ax, γ) in zip(axes, (1, 10, 20, 30))
        model = create_investment_model(γ=γ)
        (; β, a_0, a_1, γ, c, y_grid, z_grid, Q) = model
        σ_star = optimistic_policy_iteration(model)
        mc = MarkovChain(Q, z_grid)

        z_sim_idx = simulate_indices(mc, ts_length)
        z_sim = z_grid[z_sim_idx]
        y_sim_idx = Vector{Int32}(undef, ts_length)
        y_1 = (a_0 - c + z_sim[1]) / (2 * a_1)
        y_sim_idx[1] = searchsortedfirst(y_grid, y_1)
        for t in 1:(ts_length-1)
            y_sim_idx[t+1] = σ_star[y_sim_idx[t], z_sim_idx[t]]
        end
        y_sim = y_grid[y_sim_idx]
        y_bar_sim = (a_0 .- c .+ z_sim) ./ (2 * a_1)

        ax.plot(1:ts_length, y_sim, label=L"Y_t")
        ax.plot(1:ts_length, y_bar_sim, label=L"\bar Y_t")
        ax.legend(fontsize=fontsize, frameon=false, loc="upper right")
        ax.set_ylabel("output", fontsize=fontsize)
        ax.set_ylim(1, 9)
        ax.set_title(L"\gamma = " * "$γ", fontsize=fontsize)
    end

    fig.tight_layout()
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


function plot_timing(; m_vals=collect(range(1, 600, step=10)),
                   savefig=false,
                   figname="./figures/finite_lq_time.pdf"
    )
    model = create_investment_model()
    #println("Running Howard policy iteration.")
    #pi_time = @elapsed σ_pi = policy_iteration(model)
    #println("PI completed in $pi_time seconds.")
    println("Running value function iteration.")
    vfi_time = @elapsed σ_vfi = value_iteration(model, tol=1e-5)
    println("VFI completed in $vfi_time seconds.")
    #@assert σ_vfi == σ_pi "Warning: policies deviated."
    opi_times = []
    for m in m_vals
        println("Running optimistic policy iteration with m=$m.")
        opi_time = @elapsed σ_opi = 
            optimistic_policy_iteration(model, m=m, tol=1e-5)
        println("OPI with m=$m completed in $opi_time seconds.")
        #@assert σ_opi == σ_pi "Warning: policies deviated."
        push!(opi_times, opi_time)
    end
    fig, ax = plt.subplots(figsize=(9, 5.2))
    #ax.plot(m_vals, fill(pi_time, length(m_vals)), 
    #        lw=2, label="Howard policy iteration")
    ax.plot(m_vals, fill(vfi_time, length(m_vals)), 
            lw=2, label="value function iteration")
    ax.plot(m_vals, opi_times, lw=2, label="optimistic policy iteration")
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_xlabel(L"m", fontsize=fontsize)
    ax.set_ylabel("time", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end
    return (vfi_time, opi_times)
    #return (pi_time, vfi_time, opi_times)
end
