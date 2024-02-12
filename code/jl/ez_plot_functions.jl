
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16


function plot_policy(σ, model; title, savefig=false, figname="policies.pdf")
    w_grid = model.w_grid
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, w_grid, "k--", label=L"45")
    ax.plot(w_grid, w_grid[σ[:, 1]], label=L"\sigma^*(\cdot, e_1)")
    ax.plot(w_grid, w_grid[σ[:, end]], label=L"\sigma^*(\cdot, e_N)")
    #ax.set_title(title, fontsize=16)
    ax.legend(fontsize=fontsize)
    if savefig
        plt.savefig(figname)
    end
    plt.show()
end

function plot_value_orig(v, model)
    w_grid = model.w_grid
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, v[:, 1], label=L"v^*(\cdot, e_1)")
    ax.plot(w_grid, v[:, end], label=L"v^*(\cdot, e_N)")
    ax.legend(fontsize=fontsize)
    plt.show()
end

function plot_value_mod(h, model)
    w_grid = model.w_grid
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, h, label=L"h^*")
    ax.legend(fontsize=fontsize)
    plt.show()
end
