import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from s_approx import successive_approx
from two_period_job_search import create_job_search_model


@njit
def g(h, model):
    n, w_vals, ϕ, β, c = model
    return c + β * np.dot(np.maximum(w_vals / (1 - β), h), ϕ)

def compute_hstar_wstar(model, h_init=0.0):
    n, w_vals, ϕ, β, c = model
    h_star = successive_approx(lambda h: g(h, model), h_init)
    return h_star, (1 - β) * h_star

# Plot functions
def fig_g(model, savefig=False, fs=16,
          figname="../figures_py/iid_job_search_g.png"):
    n, w_vals, ϕ, β, c = model
    h_grid = np.linspace(600, max(c, n) / (1 - β), 100)
    g_vals = np.array([g(h, model) for h in h_grid])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(h_grid, g_vals, lw=2.0, label=r"$g$")
    ax.plot(h_grid, h_grid, "k--", lw=1.0, label="45")

    ax.legend(frameon=False, fontsize=fs, loc="lower right")

    h_star, w_star = compute_hstar_wstar(model)
    ax.plot(h_star, h_star, "go", ms=10, alpha=0.6)

    ax.annotate(r"$h^*$", 
                xy=(h_star, h_star),
                xycoords="data",
                xytext=(40, -40),
                textcoords="offset points",
                fontsize=fs)

    if savefig:
        fig.savefig(figname)

    plt.show()

def fig_tg(betas=[0.95, 0.96], savefig=False, fs=16,
           figname="../figures_py/iid_job_search_tg.png"):
    h_grid = np.linspace(600, 1200, 100)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(h_grid, h_grid, "k--", lw=1.0, label="45")

    for i, β in enumerate(betas):
        model = create_job_search_model(β=β)
        n, w_vals, ϕ, β, c = model
        g_vals = np.array([g(h, model) for h in h_grid])

        lb = f"$g_{i+1} \\; (\\beta_{i+1} = {β})$"
        ax.plot(h_grid, g_vals, lw=2.0, label=lb)

        ax.legend(frameon=False, fontsize=fs, loc="lower right")

        h_star, w_star = compute_hstar_wstar(model)
        ax.plot(h_star, h_star, "go", ms=10, alpha=0.6)

        lb = f"$h^*_{i+1}$"
        ax.annotate(lb, 
                    xy=(h_star, h_star),
                    xycoords="data",
                    xytext=(40, -40),
                    textcoords="offset points",
                    fontsize=fs)

    if savefig:
        fig.savefig(figname)

    plt.show()

def fig_cv(model, fs=16, savefig=False,
           figname="../figures_py/iid_job_search_4.png"):
    n, w_vals, ϕ, β, c = model
    h_star, w_star = compute_hstar_wstar(model)
    vhat = np.maximum(w_vals / (1 - β), h_star)

    fig, ax = plt.subplots()
    ax.plot(w_vals, vhat, "k-", lw=2.0, label="value function")
    ax.legend(fontsize=fs)
    ax.set_ylim(0, np.max(vhat))

    plt.show()
    if savefig:
        fig.savefig(figname)

def fig_bf(betas=np.linspace(0.9, 0.99, 20), savefig=False, fs=16,
           figname="../figures_py/iid_job_search_bf.png"):
    h_vals = np.zeros_like(betas)
    for i, β in enumerate(betas):
        model = create_job_search_model(β=β)
        h, w = compute_hstar_wstar(model)
        h_vals[i] = h

    fig, ax = plt.subplots()
    ax.plot(betas, h_vals, lw=2.0, alpha=0.7, label=r"$h^*(\beta)$")
    ax.legend(frameon=False, fontsize=fs)
    ax.set_xlabel(r"$\beta$", fontsize=fs)
    ax.set_ylabel("continuation value", fontsize=fs)

    if savefig:
        fig.savefig(figname)

    plt.show()

default_model = create_job_search_model()

fig_g(default_model, savefig=True)
fig_tg(savefig=True)
fig_cv(default_model, savefig=True)
fig_bf(savefig=True)
