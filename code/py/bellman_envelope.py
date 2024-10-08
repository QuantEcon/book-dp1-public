import numpy as np
import matplotlib.pyplot as plt


xmin = -0.5
xmax = 2.0

xgrid = np.linspace(xmin, xmax, 1000)

a1, b1 = 0.15, 0.5    # first T_σ
a2, b2 = 0.5, 0.4     # second T_σ
a3, b3 = 0.75, 0.2    # third T_σ

v1 = b1/(1-a1)
v2 = b2/(1-a2)
v3 = b3/(1-a3)

T1 = a1 * xgrid + b1
T2 = a2 * xgrid + b2
T3 = a3 * xgrid + b3
T =  np.maximum.reduce([T1, T2, T3])

fig, ax = plt.subplots(figsize=(6, 5))
for spine in ["left", "bottom"]:
    ax.spines[spine].set_position("zero")

for spine in ["right", "top"]:
    ax.spines[spine].set_color("none")

ax.plot(xgrid, T1, "k-", lw=1)
ax.plot(xgrid, T2, "k-", lw=1)
ax.plot(xgrid, T3, "k-", lw=1)

ax.plot(xgrid, T, lw=6, alpha=0.3, color="blue",
        label=r"$T = \bigvee_{\sigma \in \Sigma} T_\sigma$")


ax.text(2.1, 0.6, r"$T_{\sigma'}$")
ax.text(2.1, 1.4, r"$T_{\sigma''}$")
ax.text(2.1, 1.9, r"$T_{\sigma'''}$")

ax.legend(frameon=False, loc="upper center")


ax.set_xlim(xmin, xmax+0.5)
ax.set_ylim(-0.2, 2.5)
ax.text(2.4, -0.15, r"$v$")

ax.set_xticks([])
ax.set_yticks([])

plt.show()

file_name = "../figures_py/bellman_envelope.png"
fig.savefig(file_name)

