import numpy as np
import matplotlib.pyplot as plt

fontsize = 16

def subplots():
    """Custom subplots with axes through the origin"""
    fig, ax = plt.subplots()

    # Set the axes through the origin
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_position("zero")
    for spine in ["right", "top"]:
        ax.spines[spine].set_color("none")

    return fig, ax

x_grid = np.linspace(0, 2, 200)

# Function
def g(x):
    return np.sqrt(x)

# First order approximation at 1
def hat_g(x):
    return 1 + (1/2) * (x - 1)

fig, ax = subplots()
ax.plot(x_grid, x_grid, 'k--', label=r"$45^\circ$")
ax.plot(x_grid, g(x_grid), label=r"$g$")
ax.plot(x_grid, hat_g(x_grid), label=r"$\hat g$")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="upper center", fontsize=16)

fp = [1]
ax.plot(fp, fp, 'ko', ms=8, alpha=0.6)

ax.annotate(r"$x^*$", 
            xy=(1, 1),
            xycoords="data",
            xytext=(40, -80),
            textcoords="offset points",
            fontsize=16,
            arrowprops=dict(arrowstyle="->"))

plt.savefig("../figures_py/loc_stable_1.png")
plt.show()
