using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

x0 = 0.25
xmin, xmax = 0, 3
fs = 18

x_grid = LinRange(xmin, xmax, 1200)

g(x) = 1 + 0.5 * x^0.5 
xstar = 1.64

fig, ax = plt.subplots(figsize=(10, 5.5))
# Plot the functions
lb = L"g"
ax.plot(x_grid, g.(x_grid),  lw=2, alpha=0.6, label=lb)
ax.plot(x_grid, x_grid, "k--", lw=1, alpha=0.7, label=L"45")

# Show and annotate the fixed point
fps = (xstar,)
ax.plot(fps, fps, "go", ms=10, alpha=0.6)
ax.annotate(L"x^*", 
         xy=(xstar, xstar),
         xycoords="data",
         xytext=(-20, 20),
         textcoords="offset points",
         fontsize=fs)
         #arrowstyle="->")
    
ax.legend(loc="upper left", frameon=false, fontsize=fs)
ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((0, 1, 2, 3))
ax.set_ylim(0, 3)
ax.set_xlim(0, 3)

plt.show()
#fig.savefig("../figures/concave_map_fp.pdf")


