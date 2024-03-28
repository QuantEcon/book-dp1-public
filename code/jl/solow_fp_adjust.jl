using PyPlot, LinearAlgebra
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

function subplots(fs)
    "Custom subplots with axes throught the origin"
    fig, ax = plt.subplots(2, 2, figsize=fs)

    for i in 1:2
        for j in 1:2
            # Set the axes through the origin
            for spine in ["left", "bottom"]
                ax[i,j].spines[spine].set_position("zero")
                ax[i,j].spines[spine].set_color("black")
            end
            for spine in ["right", "top"]
                ax[i,j].spines[spine].set_color("none")
            end
        end
    end

    return fig, ax
end

# Create parameter sets where we adjust A, s, and delta
A = [2, 2.5, 2, 2]
s = [.3, .3, .2, .3]
alpha = [.3, .3, .3, .3]
delta = [.4, .4, .4, .6]
x0 = 0.25
num_arrows = 8
ts_length = 12
xmin, xmax = 0, 3
g(k, s, A, delta, alpha) = A * s * k^alpha + (1 - delta) * k
kstar(s, A, delta, alpha) = ((s * A) / delta)^(1/(1 - alpha))

xgrid = LinRange(xmin, xmax, 120)

fig, ax = subplots((10, 7))
# (0,0) is the default parameters
# (0,1) increases A
# (1,0) decreases s
# (1,1) increases delta

lb = ["default", L"$A=2.5$", L"$s=.2$", L"$\delta=.6$"]
count = 1
for i in 1:2
    for j in 1:2
        ax[i,j].set_xlim(xmin, xmax)
        ax[i,j].set_ylim(xmin, xmax)
        ax[i,j].plot(xgrid, g.(xgrid, s[count], A[count], delta[count], alpha[count]), 
                       "b-", lw=2, alpha=0.6, label=lb[count])
        ks = kstar(s[count], A[count], delta[count], alpha[count])
        ax[i,j].plot(ks, ks, "go")
        ax[i,j].plot(xgrid, xgrid, "k-", lw=1, alpha=0.7)
        global count += 1
        ax[i,j].legend(loc="lower right", frameon=false, fontsize=14)
    end
end

plt.show()

fig.savefig("../figures/solow_fp_adjust.pdf")


