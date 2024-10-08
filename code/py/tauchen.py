import numpy as np
import matplotlib.pyplot as plt
from quantecon.markov import tauchen

ρ, b, ν = 0.9, 0.0, 1.0
μ_x = b / (1 - ρ)
σ_x = np.sqrt(ν**2 / (1.0 - ρ**2))

# Number of states
n = 15

mc = tauchen(n, ρ, ν)
approx_sd = mc.stationary_distributions[0]

def psi_star(y):
    c = 1 / (np.sqrt(2 * np.pi) * σ_x)
    return c * np.exp(-(y - μ_x)**2 / (2 * σ_x**2))

# == Plots == #
fontsize = 10

fig, ax = plt.subplots()

ax.bar(mc.state_values, approx_sd, width=0.6, alpha=0.6, label="approximation")

x_grid = np.linspace(min(mc.state_values) - 2, max(mc.state_values) + 2, 100)
ax.plot(x_grid, psi_star(x_grid), '-k', lw=2, alpha=0.6, label=r"$\psi^*$")

ax.legend(fontsize=fontsize)

plt.show()

plt.savefig("../figures_py/tauchen_1.png")
