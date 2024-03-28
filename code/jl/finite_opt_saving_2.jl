include("s_approx.jl")
include("finite_opt_saving_1.jl")

"Value function iteration routine."
function value_iteration(model, tol=1e-5)
    vz = zeros(length(model.w_grid), length(model.y_grid))
    v_star = successive_approx(v -> T(v, model), vz, tolerance=tol)
    return get_greedy(v_star, model)
end

"Howard policy iteration routine."
function policy_iteration(model)
    wn, yn = length(model.w_grid), length(model.y_grid)
    σ = ones(Int32, wn, yn)
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
function optimistic_policy_iteration(model; tolerance=1e-5, m=100)
    v = zeros(length(model.w_grid), length(model.y_grid))
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

# == Simulations and inequality measures == #

function simulate_wealth(m)

    model = create_savings_model()
    σ_star = optimistic_policy_iteration(model)
    (; β, R, γ, w_grid, y_grid, Q) = model

    # Simulate labor income (indices rather than grid values)
    mc = MarkovChain(Q)
    y_idx_series = simulate(mc, m)

    # Compute corresponding wealth time series
    w_idx_series = similar(y_idx_series)
    w_idx_series[1] = 1  # initial condition
    for t in 1:(m-1)
        i, j = w_idx_series[t], y_idx_series[t]
        w_idx_series[t+1] = σ_star[i, j]
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

function plot_timing(; m_vals=collect(range(1, 600, step=10)),
                       savefig=false)
    model = create_savings_model(y_size=5)
    println("Running Howard policy iteration.")
    pi_time = @elapsed σ_pi = policy_iteration(model)
    println("PI completed in $pi_time seconds.")
    println("Running value function iteration.")
    vfi_time = @elapsed σ_vfi = value_iteration(model)
    println("VFI completed in $vfi_time seconds.")
    @assert σ_vfi == σ_pi "Warning: policies deviated."
    opi_times = []
    for m in m_vals
        println("Running optimistic policy iteration with m=$m.")
        opi_time = @elapsed σ_opi = optimistic_policy_iteration(model, m=m)
        @assert σ_opi == σ_pi "Warning: policies deviated."
        println("OPI with m=$m completed in $opi_time seconds.")
        push!(opi_times, opi_time)
    end
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(m_vals, fill(vfi_time, length(m_vals)), 
            lw=2, label="value function iteration")
    ax.plot(m_vals, fill(pi_time, length(m_vals)), 
            lw=2, label="Howard policy iteration")
    ax.plot(m_vals, opi_times, lw=2, label="optimistic policy iteration")
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_xlabel(L"m", fontsize=fontsize)
    ax.set_ylabel("time", fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig("../figures/finite_opt_saving_2_1.pdf")
    end
    return (pi_time, vfi_time, opi_times)
end

function plot_policy(; method="pi")
    model = create_savings_model()
    (; β, R, γ, w_grid, y_grid, Q) = model
    if method == "vfi"
        σ_star =  value_iteration(model) 
    elseif method == "pi"
        σ_star = policy_iteration(model) 
    else
        σ_star = optimistic_policy_iteration(model)
    end
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, w_grid, "k--", label=L"45")
    ax.plot(w_grid, w_grid[σ_star[:, 1]], label=L"\sigma^*(\cdot, y_1)")
    ax.plot(w_grid, w_grid[σ_star[:, end]], label=L"\sigma^*(\cdot, y_N)")
    ax.legend(fontsize=fontsize)
    plt.show()
end


function plot_time_series(; m=2_000,
                           savefig=false, 
                           figname="../figures/finite_opt_saving_ts.pdf")

    w_series = simulate_wealth(m)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_series, label=L"w_t")
    ax.set_xlabel("time", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

function plot_histogram(; m=1_000_000,
                           savefig=false, 
                           figname="../figures/finite_opt_saving_hist.pdf")

    w_series = simulate_wealth(m)
    g = round(gini(sort(w_series)), digits=2)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(w_series, bins=40, density=true)
    ax.set_xlabel("wealth", fontsize=fontsize)
    ax.text(15, 0.4, "Gini = $g", fontsize=fontsize)
    plt.show()

    if savefig
        fig.savefig(figname)
    end
end

function plot_lorenz(; m=1_000_000,
                           savefig=false, 
                           figname="../figures/finite_opt_saving_lorenz.pdf")

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
