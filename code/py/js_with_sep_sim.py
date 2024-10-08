import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from markov_js_with_sep import create_js_with_sep_model, vfi
from numba import njit

# Create and solve model
model = create_js_with_sep_model()
n, w_vals, P, β, c, α = model
v_star, σ_star = vfi(model)

# Create Markov distributions to draw from
P_dists = [rv_discrete(values=(np.arange(n), P[i, :])) for i in range(n)]

def update_wages_idx(w_idx):
    return P_dists[w_idx].rvs()

@njit
def sim_wages(ts_length=100):
    w_idx = np.random.randint(0, n-1)
    W = np.zeros(ts_length)
    for t in range(ts_length):
        W[t] = w_vals[w_idx]
        w_idx = update_wages_idx(w_idx)
    return W

def sim_outcomes(ts_length=100):
    status = 0
    E, W = [], []
    w_idx = np.random.randint(0, n-1)
    for t in range(ts_length):
        if status == 0:
            status = σ_star[w_idx] if σ_star[w_idx] else 0
        else:
            status = 0 if np.random.rand() < α else 1
        W.append(w_vals[w_idx])
        E.append(status)
        w_idx = update_wages_idx(w_idx)
    return W, E

# == Plots == #

def plot_status(ts_length=100, savefig=False,
                figname="../figures_py/js_with_sep_sim_1.png"):
    W, E = sim_outcomes(ts_length=ts_length)
    fs = 16
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    ax.plot(W, label="wage offers")
    ax.legend(fontsize=fs, frameon=False)

    ax = axes[1]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["unempl.", "employed"])
    ax.plot(E, label="status")
    ax.legend(fontsize=fs, frameon=False)

    if savefig:
        fig.savefig(figname)
    plt.show()

plot_status(savefig=True)
