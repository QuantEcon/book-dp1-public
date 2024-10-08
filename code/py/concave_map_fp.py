import numpy as np
import matplotlib.pyplot as plt

x0 = 0.25
xmin, xmax = 0, 3
fs = 18

x_grid = np.linspace(xmin, xmax, 1200)

def g(x):
    return 1 + 0.5 * x**0.5

xstar = 1.64

fig, ax = plt.subplots(figsize=(10, 5.5))

# Plot the functions
lb = r"$g$"
ax.plot(x_grid, g(x_grid), lw=2, alpha=0.6, label=lb)
ax.plot(x_grid, x_grid, "k--", lw=1, alpha=0.7, label=r"$45$")

# Show and annotate the fixed point
fps = (xstar,)
ax.plot(fps, fps, "go", ms=10, alpha=0.6)
ax.set_xlabel(r"$x$", fontsize=fs)
ax.annotate(r"$x^*$",
            xy=(xstar, xstar),
            xycoords="data",
            xytext=(-20, 20),
            textcoords="offset points",
            fontsize=fs)

ax.legend(loc="upper left", frameon=False, fontsize=fs)
ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((0, 1, 2, 3))
ax.set_ylim(0, 3)
ax.set_xlim(0, 3)

plt.show()

file_name = "../figures_py/concave_map_fp.png"
fig.savefig(file_name)
