import matplotlib.pyplot as plt
import numpy as np


def subplots(fs):
    """Custom subplots with axes through the origin"""
    fig, ax = plt.subplots(2, 2, figsize=fs)

    for i in range(2):
        for j in range(2):
            # Set the axes through the origin
            for spine in ["left", "bottom"]:
                ax[i, j].spines[spine].set_position("zero")
                ax[i, j].spines[spine].set_color("black")
            for spine in ["right", "top"]:
                ax[i, j].spines[spine].set_color("none")
    
    return fig, ax

A = [2, 2.5, 2, 2]
s = [0.3, 0.3, 0.2, 0.3]
alpha = [0.3, 0.3, 0.3, 0.3]
delta = [0.4, 0.4, 0.4, 0.6]
x0 = 0.25
num_arrows = 8
ts_length = 12
xmin, xmax = 0, 3

def g(k, s, A, delta, alpha):
    return A * s * k**alpha + (1 - delta) * k

def kstar(s, A, delta, alpha):
    return ((s * A) / delta)**(1/(1 - alpha))

xgrid = np.linspace(xmin, xmax, 120)

fig, ax = subplots((10, 7))

# (0,0) is the default parameters
# (0,1) increases A
# (1,0) decreases s
# (1,1) increases delta

lb = ["default", r"$A=2.5$", r"$s=.2$", r"$\delta=.6$"]
count = 0

for i in range(2):
    for j in range(2):
        ax[i, j].set_xlim(xmin, xmax)
        ax[i, j].set_ylim(xmin, xmax)
        ax[i, j].plot(xgrid, g(xgrid, s[count], A[count], delta[count], alpha[count]), 
                      "b-", lw=2, alpha=0.6, label=lb[count])
        ks = kstar(s[count], A[count], delta[count], alpha[count])
        ax[i, j].plot(ks, ks, "go")
        ax[i, j].plot(xgrid, xgrid, "k-", lw=1, alpha=0.7)
        count += 1
        ax[i, j].legend(loc="lower right", frameon=False, fontsize=14)

plt.show()

file_name = "../figures_py/solow_fp_adjust.png"
fig.savefig(file_name)
