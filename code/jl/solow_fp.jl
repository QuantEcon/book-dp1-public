using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

x0 = 0.25
xmin, xmax = 0, 3


k_grid = LinRange(xmin, xmax, 1200)

function plot_45(ax; k0=0.5, 
                A=2.0, s=0.3, alpha=0.3, delta=0.4,
                fs=16, # font size
                num_arrows=8)

    # Define the function and the fixed point
    g(k) = A * s * k^alpha + (1 - delta) * k
    kstar = ((s * A) / delta)^(1/(1 - alpha))

    # Plot the functions
    lb = L"g(k) = sAk^{\alpha} + (1 - \delta)k"
    ax.plot(k_grid, g.(k_grid),  lw=2, alpha=0.6, label=lb)
    ax.plot(k_grid, k_grid, "k--", lw=1, alpha=0.7, label=L"45")

    # Show and annotate the fixed point
    fps = (kstar,)
    ax.plot(fps, fps, "go", ms=10, alpha=0.6)
    ax.annotate(L"k^* = (sA / \delta)^{\frac{1}{1-\alpha}}", 
             xy=(kstar, kstar),
             xycoords="data",
             xytext=(20, -20),
             textcoords="offset points",
             fontsize=fs)
             #arrowstyle="->")
    
    # Draw the arrow sequence

    arrow_args = Dict(:fc=>"k", :ec=>"k", :head_width=>0.03,
            :length_includes_head=>true, :lw=>1,
            :alpha=>0.6, :head_length=>0.03)

    k = k0
    for i in 1:num_arrows
        ax.arrow(k, k, 0.0, g(k)-k; arrow_args...) # x, y, dx, dy
        ax.arrow(k, g(k), g(k) - k, 0; arrow_args...)
        k = g(k)
    end


    ax.legend(loc="upper left", frameon=false, fontsize=fs)

    ax.set_xticks((0, k0, 3))
    ax.set_xticklabels((0, L"k_0", 3), fontsize=fs)
    ax.set_yticks((0, 1, 2, 3))
    ax.set_yticklabels((0, 1, 2, 3), fontsize=fs)
    ax.set_ylim(0, 3)
    ax.set_xlabel(L"k_t", fontsize=fs)
    ax.set_ylabel(L"k_{t+1}", fontsize=fs)

end



fig, ax = plt.subplots()

plot_45(ax; A=2.0, s=0.3, alpha=0.4, delta=0.4) 
#plot_45(ax2; A=3.0, s=0.4, alpha=0.05, delta=0.6) 
fig.tight_layout()
plt.show()
#fig.savefig("../figures/solow_fp.pdf")


