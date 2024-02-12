using PyPlot
using LaTeXStrings
using ForwardDiff
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

A, s, alpha, delta = 2, 0.3, 0.3, 0.4
x0 = 0.25
n = 14

g(k) = A * s * k^alpha + (1 - delta) * k
Dg = x -> ForwardDiff.derivative(g, float(x))
q(x) = (g(x) - Dg(x) * x) / (1 - Dg(x))

fs = 14
kstar = ((s * A) / delta)^(1/(1 - alpha))

function plot_45(; file_name="./figures/newton_solow_45.pdf",
                 xmin=0.0, xmax=4,
                 save_fig=false)

    xgrid = LinRange(xmin, xmax, 1200)

    fig, ax = plt.subplots()

    lb_g = L"g"
    ax.plot(xgrid, g.(xgrid),  lw=2, alpha=0.6, label=lb_g)

    lb_q = L"Q"
    ax.plot(xgrid, q.(xgrid),  lw=2, alpha=0.6, label=lb_q)

    ax.plot(xgrid, xgrid, "k--", lw=1, alpha=0.7, label=L"45")

    fps = (kstar,)
    ax.plot(fps, fps, "go", ms=10, alpha=0.6)

    # ax.annotate(L"k^* = (sA / \delta)^{\frac{1}{1-\alpha}}", 
             # xy=(kstar, kstar),
             # xycoords="data",
             # xytext=(20, -20),
             # textcoords="offset points",
             # fontsize=14)
             # #arrowstyle="->")

    ax.legend(frameon=false, fontsize=14)

    #ax.set_xticks((0, 1, 2, 3))
    #ax.set_yticks((0, 1, 2, 3))

    ax.set_xlabel(L"k_t", fontsize=14)
    ax.set_ylabel(L"k_{t+1}", fontsize=14)

    ax.set_ylim(-3, 4)
    ax.set_xlim(0, 4)

    plt.show()
    if save_fig
        fig.savefig(file_name)
    end
end


function compute_iterates(k0, f)
    k = k0
    k_iterates = []
    for t in 1:n
        push!(k_iterates, k)
        k = f(k)
    end
    return k_iterates
end



function plot_trajectories(; file_name="./figures/newton_solow_traj.pdf",
                           save_fig=false)

    x_grid = collect(1:n)

    fig, axes = plt.subplots(2, 1)
    ax1, ax2 = axes

    k0_a, k0_b = 0.8, 3.1

    ks1 = compute_iterates(k0_a, g)
    ax1.plot(x_grid, ks1, "-o", label="successive approximation")

    ks2 = compute_iterates(k0_b, g)
    ax2.plot(x_grid, ks2, "-o", label="successive approximation")

    ks3 = compute_iterates(k0_a, q)
    ax1.plot(x_grid, ks3, "-o", label="newton steps")

    ks4 = compute_iterates(k0_b, q)
    ax2.plot(x_grid, ks4, "-o", label="newton steps")


    for ax in axes
        ax.plot(x_grid, kstar * ones(n), "k--")
        ax.legend(fontsize=fs, frameon=false)
        ax.set_ylim(0.6, 3.2)
        xticks = (2, 4, 6, 8, 10, 12)
        ax.set_xticks(xticks)
        ax.set_xticklabels([string(s) for s in xticks], fontsize=fs)
        ax.set_yticks((kstar,))
        ax.set_yticklabels((L"k^*",), fontsize=fs)
    end


    plt.show()
    if save_fig
        fig.savefig(file_name)
    end
end







