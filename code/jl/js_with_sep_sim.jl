"""
Simulation of the job search with Markov wage draws and separation.

"""

include("markov_js_with_sep.jl")   # Code to solve model
using Distributions

# Create and solve model
model = create_js_with_sep_model()
(; n, w_vals, P, β, c, α) = model
v_star, σ_star = vfi(model)

# Create Markov distributions to draw from
P_dists = [DiscreteRV(P[i, :]) for i in 1:n]

function update_wages_idx(w_idx)
    return rand(P_dists[w_idx])
end

function sim_wages(ts_length=100)
    w_idx = rand(DiscreteUniform(1, n))
    W = zeros(ts_length)
    for t in 1:ts_length
        W[t] = w_vals[w_idx]
        w_idx = update_wages_idx(w_idx)
    end
    return W
end

function sim_outcomes(; ts_length=100)
    status = 0
    E, W = [], []
    w_idx = rand(DiscreteUniform(1, n))
    ts_length = 100
    for t in 1:ts_length
        if status == 0
            status = σ_star[w_idx] ? 1 : 0
        else
            status = rand() < α ? 0 : 1
        end
        push!(W, w_vals[w_idx])
        push!(E, status)
        w_idx = update_wages_idx(w_idx)
    end
    return W, E
end


# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

function plot_status(; ts_length=200,
                      savefig=false, 
                      figname="../figures/js_with_sep_sim_1.pdf")

    W, E = sim_outcomes()
    fs = 16
    fig, axes = plt.subplots(2, 1)

    ax = axes[1]
    ax.plot(W, label="wage offers")
    ax.legend(fontsize=fs, frameon=false)

    ax = axes[2]
    ax.set_yticks((0, 1))
    ax.set_yticklabels(("unempl.", "employed"))
    ax.plot(E, label="status")
    ax.legend(fontsize=fs, frameon=false)

    plt.show()
    if savefig
        fig.savefig(figname)
    end

end
