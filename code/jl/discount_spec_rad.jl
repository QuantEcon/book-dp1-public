using LinearAlgebra                         
using PyPlot
using LaTeXStrings
using QuantEcon
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=18

ρ(A) = maximum(abs(λ) for λ in eigvals(A))  # Spectral radius

function plot_contours(; savefig=false, 
                         figname="../figures/discount_spec_rad.pdf")

    fig, ax = plt.subplots(figsize=(8,4.2))
    grid_size = 50  
    n = 6
    a_vals = LinRange(0.1, 0.95, grid_size)
    s_vals = LinRange(0.1, 0.32, grid_size)
    μ = 0.96

    R = zeros(grid_size, grid_size)
    L = zeros(n, n)

    for (i_a, a) in enumerate(a_vals)
        for (i_s, s) in enumerate(s_vals)
            mc = tauchen(n, a, sqrt(1 - a^2) * s, (1 - a) * μ)
            z_vals, Q = mc.state_values, mc.p
            β_vals = z_vals # max.(z_vals, 0)
            println(minimum(z_vals))
            for i in 1:n
                for j in 1:n
                    L[i, j] = β_vals[i] * Q[i, j]
                end
            end
            R[i_a, i_s] = ρ(L)
        end
    end

    cs1 = ax.contourf(a_vals, s_vals, transpose(R), alpha=0.5)
    ctr1 = ax.contour(a_vals, s_vals, transpose(R), levels=[1.0])
    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax) #, format="%.6f")

    ax.set_xlabel(L"a", fontsize=fontsize)
    ax.set_ylabel(L"s", fontsize=fontsize)

    #ax.set_xticks((minimum(a_vals), maximum(a_vals)))
    #ax.set_yticks((minimum(s_vals), maximum(s_vals)))

    if savefig
        fig.savefig(figname)
    end
    plt.show()
end


