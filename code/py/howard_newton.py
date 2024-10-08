import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from autograd import grad

v0 = 0.5
T = lambda x: 1 + 0.2 * x**2
DT = grad(T)
Tsp = lambda v, v0=v0: T(v0) + DT(v0) * (v - v0)  # T_\sigma'

vs = newton(lambda v: T(v) - v, 0.5)              # find fixed point of T
v1 = (T(v0) - DT(v0) * v0) / (1 - DT(v0))         # v_\sigma'

fs = 16

xmin, xmax = -0.2, 2.5
xgrid = np.linspace(xmin, xmax, 1000)

fig, ax = plt.subplots()

lb_T = r"$T$"
ax.plot(xgrid, T(xgrid), lw=2, alpha=0.6, label=lb_T)

lb_Tsp = r"$T_{\sigma'}$"
ax.plot(xgrid, Tsp(xgrid), lw=2, alpha=0.6, label=lb_Tsp)

ax.plot(xgrid, xgrid, "k--", lw=1, alpha=0.7, label=r"$45^{\circ}$")

fp1 = [v1]
ax.plot(fp1, fp1, "go", ms=5, alpha=0.6)
ax.plot([v0], [Tsp(v0)], "go", ms=5, alpha=0.6)
ax.plot([vs], [vs], "go", ms=5, alpha=0.6)

ax.vlines([v0, vs, v1], [0, 0, 0], [Tsp(v0), vs, v1], 
          color="k", linestyle="-.", lw=0.4)

ax.legend(frameon=False, fontsize=fs)

ax.set_xticks([v0, vs, v1])
ax.set_xticklabels([r"$v_\sigma$", r"$v^*$", r"$v_{\sigma'}$"], fontsize=fs)
ax.set_yticks([0])

ax.set_xlim(0, 2.6)
ax.set_ylim(0, 2.6)

plt.show()

filename = "../figures_py/howard_newton_1.png"

fig.savefig(filename)
