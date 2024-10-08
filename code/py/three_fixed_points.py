import numpy as np
import matplotlib.pyplot as plt

fs = 14
xmin, xmax = 0.0000001, 2.0

def g(u):
    return 2.125 / (1 + u**-4)

xgrid = np.linspace(xmin, xmax, 200)

fig, ax = plt.subplots(figsize=(6.5, 6))

ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

ax.plot(xgrid, g(xgrid), "b-", lw=2, alpha=0.6, label=r"$T$")
ax.plot(xgrid, xgrid, "k-", lw=1, alpha=0.7, label=r"$45^\circ$")

ax.legend(fontsize=fs)

fps = [0.01, 0.94, 1.98]
fps_labels = [r"$u_\ell$", r"$u_m$", r"$u_h$"]
coords = [(40, 80), (40, -40), (-40, -80)]

ax.plot(fps, fps, "ro", ms=8, alpha=0.6)

for fp, lb, coord in zip(fps, fps_labels, coords):
    ax.annotate(lb,
                xy=(fp, fp),
                xycoords="data",
                xytext=coord,
                textcoords="offset points",
                fontsize=fs,
                arrowprops=dict(arrowstyle="->"))

plt.savefig("../figures_py/three_fixed_points.png")
plt.show()
