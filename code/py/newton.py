import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from autograd import grad

fs = 16

# Define the function T and its derivative
def T(x):
    return 1 + (x / (x + 1))

# Autograd to compute the derivative of T
DT = grad(T)

x0 = 0.5

def T_hat(x, x0=x0):
    return T(x0) + DT(x0) * (x - x0)

# Find the fixed point of T
res = root_scalar(lambda x: T(x) - x, x0=0.5)
xs = res.root

# Compute the Newton approximation x1
x1 = (T(x0) - DT(x0) * x0) / (1 - DT(x0))

# Plot
def plot_45(file_name="../figures_py/newton_1.png", xmin=0.0, xmax=2.6,
            savefig=False):
    xgrid = np.linspace(xmin, xmax, 1000)

    fig, ax = plt.subplots()

    lb_T = r"$T$"
    ax.plot(xgrid, T(xgrid), lw=2, alpha=0.6, label=lb_T)

    lb_T_hat = r"$\hat{T}$"
    ax.plot(xgrid, T_hat(xgrid), lw=2, alpha=0.6, label=lb_T_hat)

    ax.plot(xgrid, xgrid, "k--", lw=1, alpha=0.7, label=r"$45^\circ$")

    fp1 = [x1]
    ax.plot(fp1, fp1, "go", ms=5, alpha=0.6)
    ax.plot([x0], [T_hat(x0)], "go", ms=5, alpha=0.6)
    ax.plot([xs], [xs], "go", ms=5, alpha=0.6)

    ax.vlines([x0, xs, x1], [0, 0, 0], [T_hat(x0), xs, x1], 
              color="k", linestyle="-.", lw=0.4)

    ax.legend(frameon=False, fontsize=fs)

    ax.set_xticks([x0, xs, x1])
    ax.set_xticklabels([r"$u_0$", r"$u^*$", r"$u_1$"], fontsize=fs)
    ax.set_yticks([0])

    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 2.6)

    plt.show()
    if savefig:
        fig.savefig(file_name)

plot_45(savefig=True)
