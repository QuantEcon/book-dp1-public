
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

function F(w; r=0.5, β=0.5, θ=5)
    return (r + β * w^(1/θ))^θ
end

w_grid = LinRange(0.001, 2.0, 200)


function plot_F(; savefig=false, 
                  figname="./figures/ez_noncontraction.pdf",
                  fs=16)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    f(w) = F(w; θ=-10)
    ax.plot(w_grid, w_grid, "k--", alpha=0.6, label=L"45")
    ax.plot(w_grid, f.(w_grid), label=L"\hat K = F")
    ax.set_xticks((0, 1, 2))
    ax.set_yticks((0, 1, 2))
    ax.legend(fontsize=fs, frameon=false)

    plt.show()

    if savefig
        fig.savefig(figname)
    end
end



