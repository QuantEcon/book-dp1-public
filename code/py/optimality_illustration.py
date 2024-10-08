import numpy as np
import matplotlib.pyplot as plt

fs = 18

xmin = -0.5
xmax = 2.0

xgrid = np.linspace(xmin, xmax, 1000)

a1, b1 = 0.1, 0.5    # first T_σ
a2, b2 = 0.65, 0.4     # second T_σ

v1 = b1 / (1 - a1)
v2 = b2 / (1 - a2)

T1 = a1 * xgrid + b1
T2 = a2 * xgrid + b2
T = np.maximum(T1, T2)

fig, ax = plt.subplots()

# Set the axes through the origin
for spine in ["left", "bottom"]:
    ax.spines[spine].set_position("zero")
for spine in ["right", "top"]:
    ax.spines[spine].set_color("none")

ax.plot(xgrid, xgrid, "k--", lw=1, alpha=0.7, label=r"$45^{\circ}$")
ax.plot(xgrid, T1, "k-", lw=1)
ax.plot(xgrid, T2, "k-", lw=1)

ax.plot(xgrid, T, lw=6, alpha=0.3, color="blue", label=r"$T$")

ax.plot((v1,), (v1,), "go", ms=5, alpha=0.8)
ax.plot((v2,), (v2,), "go", ms=5, alpha=0.8)

ax.vlines((v1, v2), (0, 0), (v1, v2), color="k", linestyle="-.", lw=0.4)

ax.text(2.1, 0.6, r"$T_{\sigma'}$", fontsize=fs)
ax.text(2.1, 1.4, r"$T_{\sigma''}$", fontsize=fs)

ax.legend(frameon=False, loc="upper center", fontsize=fs)

ax.set_xticks([v1, v2])
ax.set_xticklabels([r"$v_{\sigma'}$", r"$v_{\sigma''} = v^*$"], fontsize=fs)
ax.set_yticks([])

ax.set_xlim(xmin, xmax + 0.5)
ax.set_ylim(-0.2, 2)
ax.text(2.4, -0.15, r"$v$", fontsize=22)

plt.show()

file_name = "../figures_py/optimality_illustration_1.png"
fig.savefig(file_name)
