import numpy as np
import matplotlib.pyplot as plt

fs = 12
xmin, xmax = 0., 1.0

def g(x):
    return 0.2 + 0.6 * x**1.2

xgrid = np.linspace(xmin, xmax, 200)

fig, ax = plt.subplots(figsize=(8.0, 6))

for spine in ["left", "bottom"]:
    ax.spines[spine].set_position("zero")
for spine in ["right", "top"]:
    ax.spines[spine].set_color("none")

ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

ax.plot(xgrid, g(xgrid), "b-", lw=2, alpha=0.6, label=r"$T$")
ax.plot(xgrid, xgrid, "k-", lw=1, alpha=0.7, label=r"$45^\circ$")

ax.legend(frameon=False, fontsize=fs)

fp = (0.4,)
fps_label = r"$\bar{v}$"
coords = (40, -20)

ax.plot(fp, fp, "ro", ms=8, alpha=0.6)

ax.annotate(fps_label, 
            xy=(fp[0], fp[0]),
            xycoords="data",
            xytext=coords,
            textcoords="offset points",
            fontsize=fs,
            arrowprops=dict(arrowstyle="->"))

ax.set_xticks([0, 1])
ax.set_xticklabels([r"$0$", r"$1$"], fontsize=fs)
ax.set_yticks([])

ax.set_xlabel(r"$V$", fontsize=fs)

plt.savefig("../figures_py/up_down_stable.png")
plt.show()
