import numpy as np
import matplotlib.pyplot as plt

fs = 12

xmin, xmax = 0.01, 2.0

xgrid = np.linspace(xmin, xmax, 1000)

v1 = xgrid ** 0.7
v2 = xgrid ** 0.1 + 0.05
v = np.maximum(v1, v2)

fig, ax = plt.subplots()

for spine in ["left", "bottom"]:
    ax.spines[spine].set_position("zero")
for spine in ["right", "top"]:
    ax.spines[spine].set_color("none")

ax.plot(xgrid, v1, "k-", lw=1)
ax.plot(xgrid, v2, "k-", lw=1)
ax.plot(xgrid, v, lw=6, alpha=0.3, color="blue", 
        label=r"$v^* = \bigvee_{\sigma \in \Sigma} v_\sigma$")

ax.text(2.1, 1.1, r"$v_{\sigma'}$", fontsize=fs)
ax.text(2.1, 1.6, r"$v_{\sigma''}$", fontsize=fs)
ax.text(1.2, 0.3, r"$\Sigma = \{\sigma', \sigma''\}$", fontsize=fs)

ax.legend(frameon=False, loc="upper left", fontsize=fs)

ax.set_xlim(xmin, xmax + 0.5)
ax.set_ylim(0.0, 2)
ax.text(2.4, -0.15, r"$x$", fontsize=20)

ax.set_xticks([])
ax.set_yticks([])

plt.show()
file_name = "../figures_py/v_star_illus.png"
fig.savefig(file_name)
