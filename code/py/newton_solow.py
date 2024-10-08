import numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad

A, s, alpha, delta = 2, 0.3, 0.3, 0.4
x0 = 0.25
n = 14

def g(k):
    return A * s * k**alpha + (1 - delta) * k

Dg = elementwise_grad(g)

def q(x):
    return (g(x) - Dg(x) * x) / (1 - Dg(x))

fs = 14
kstar = ((s * A) / delta)**(1/(1 - alpha))

def plot_45(file_name="../figures_py/newton_solow_45.png", xmin=0.0,
            xmax=4, savefig=False):
    xgrid = np.linspace(xmin, xmax, 1200)

    fig, ax = plt.subplots()

    lb_g = r"$g$"
    ax.plot(xgrid, g(xgrid), lw=2, alpha=0.6, label=lb_g)

    lb_q = r"$Q$"
    ax.plot(xgrid, q(xgrid), lw=2, alpha=0.6, label=lb_q)

    ax.plot(xgrid, xgrid, "k--", lw=1, alpha=0.7, label=r"$45^\circ$")

    fps = [kstar]
    ax.plot(fps, fps, "go", ms=10, alpha=0.6)

    ax.legend(frameon=False, fontsize=fs)

    ax.set_xlabel(r"$k_t$", fontsize=fs)
    ax.set_ylabel(r"$k_{t+1}$", fontsize=fs)

    ax.set_ylim(-3, 4)
    ax.set_xlim(0, 4)

    plt.show()
    if savefig:
        fig.savefig(file_name)


def compute_iterates(k0, f):
    k = k0
    k_iterates = []
    for _ in range(n):
        k_iterates.append(k)
        k = f(k)
    return k_iterates


def plot_trajectories(file_name="../figures_py/newton_solow_traj.png", savefig=False):
    x_grid = np.arange(1, n + 1)

    fig, axes = plt.subplots(2, 1)
    ax1, ax2 = axes

    k0_a, k0_b = 0.8, 3.1

    ks1 = compute_iterates(k0_a, g)
    ax1.plot(x_grid, ks1, "-o", label="successive approximation")

    ks2 = compute_iterates(k0_b, g)
    ax2.plot(x_grid, ks2, "-o", label="successive approximation")

    ks3 = compute_iterates(k0_a, q)
    ax1.plot(x_grid, ks3, "-o", label="newton steps")

    ks4 = compute_iterates(k0_b, q)
    ax2.plot(x_grid, ks4, "-o", label="newton steps")

    for ax in axes:
        ax.plot(x_grid, kstar * np.ones(n), "k--")
        ax.legend(fontsize=fs, frameon=False)
        ax.set_ylim(0.6, 3.2)
        xticks = [2, 4, 6, 8, 10, 12]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(s) for s in xticks], fontsize=fs)
        ax.set_yticks([kstar])
        ax.set_yticklabels([r"$k^*$"], fontsize=fs)

    plt.show()
    if savefig:
        fig.savefig(file_name)

plot_45(savefig=True)
plot_trajectories(savefig=True)
