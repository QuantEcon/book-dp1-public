using LinearAlgebra, QuantEcon

function compute_v(; n=25,       # Size of state space
                     β=0.98,     # Time discount factor
                     ρ=0.96,     # Correlation coef in AR(1)
                     ν=0.05,     # Volatility
                     γ=2.0)      # Preference parameter
    mc = tauchen(n, ρ, ν)
    x_vals = mc.state_values
    P = mc.p 
    r = exp.((1 - γ) * x_vals) / (1 - γ)  # r(x) = u(exp(x))
    v = (I - β*P) \ r  
    return x_vals, v
end

# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

function plot_v(; savefig=false, 
                  figname="../figures/val_consumption_1.pdf")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x_vals, v = compute_v()
    ax.plot(x_vals, v, lw=2, alpha=0.7, label=L"v")
    ax.set_xlabel(L"x", fontsize=fontsize)
    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    ax.set_ylim(-65, -40)
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

