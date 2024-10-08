import numpy as np
import matplotlib.pyplot as plt
from quantecon.markov import tauchen

n = 25
nu = 1.0
a = 0.5

# Generating Markov Chain using Tauchen method
mc = tauchen(n, a, nu)
state_values, P = mc.state_values, mc.P
i, j = 8, 12

fig, axes = plt.subplots(2, 1, figsize=(10, 5.2))
fontsize = 16

# Plotting the first subplot
ax = axes[0]
ax.plot(state_values, P[i, :], "b-o", alpha=0.4, lw=2, label=r"$\varphi$")
ax.plot(state_values, P[j, :], "g-o", alpha=0.4, lw=2, label=r"$\psi$")
ax.legend(frameon=False, fontsize=fontsize)

# Plotting the second subplot
ax = axes[1]
F = [np.sum(P[i, k:]) for k in range(n)]
G = [np.sum(P[j, k:]) for k in range(n)]
ax.plot(state_values, F, "b-o", alpha=0.4, lw=2, label=r"$G^\varphi$")
ax.plot(state_values, G, "g-o", alpha=0.4, lw=2, label=r"$G^\psi$")
ax.legend(frameon=False, fontsize=fontsize)

plt.show()
fig.savefig("../figures_py/fosd_tauchen_1.png")
