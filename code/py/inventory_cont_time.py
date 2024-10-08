import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, geom

def sim_path(T=10, seed=123, λ=0.5, α=0.7, b=10):
    J, Y = 0.0, b
    J_vals, Y_vals = [J], [Y]
    np.random.seed(seed)
    
    # Wait times are exponential
    φ = expon(scale=1/λ)     
    # Orders are geometric
    G = geom(p=α)         

    while True:
        W = φ.rvs()
        J += W
        J_vals.append(J)
        if Y == 0:
            Y = b
        else:
            U = G.rvs() + 1  # Geometric on 1, 2,...
            Y = Y - min(Y, U)
        Y_vals.append(Y)
        if J > T:
            break

    def X(t):
        k = np.searchsorted(J_vals, t, side='right') - 1
        return Y_vals[k+1]

    return X

T = 50
X = sim_path(T=T)

# Plotting
grid = np.linspace(0, T - 0.001, 100)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.step(grid, [X(t) for t in grid], label=r"$X_t$", alpha=0.7)

ax.set_xticks([0, 10, 20, 30, 40, 50])

ax.set_xlabel("time", fontsize=12) 
ax.set_ylabel("inventory", fontsize=12)
ax.legend(fontsize=12)

plt.savefig("../figures_py/inventory_cont_time_1.png")
plt.show()
