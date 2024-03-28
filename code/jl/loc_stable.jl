
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

function subplots()
    "Custom subplots with axes through the origin"
    fig, ax = plt.subplots()

    # Set the axes through the origin
    for spine in ["left", "bottom"]
        ax.spines[spine].set_position("zero")
    end
    for spine in ["right", "top"]
        ax.spines[spine].set_color("none")
    end

    return fig, ax
end

x_grid = LinRange(0, 2, 200)

# Function
g(x) = sqrt(x)
# First order approximation at 1
hat_g(x) = 1 + (1/2) * (x - 1)

fig, ax = subplots()
ax.plot(x_grid, x_grid, "k--", label=L"45^\circ")
ax.plot(x_grid, g.(x_grid), label=L"g")
ax.plot(x_grid, hat_g.(x_grid), label=L"\hat g")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="upper center", fontsize=16)

fp = (1,)
ax.plot(fp, fp, "ko", ms=8, alpha=0.6)


ax.annotate(L"x^*", 
         xy=(1, 1),
         xycoords="data",
         xytext=(40, -80),
         textcoords="offset points",
         fontsize=16,
         arrowprops=Dict("arrowstyle"=>"->"))

plt.savefig("../figures/loc_stable_1.pdf")
plt.show()
