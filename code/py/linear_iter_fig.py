import matplotlib.pyplot as plt
import numpy as np

from linear_iter import x_star, T

def plot_main(savefig=False, figname="./figures/linear_iter_fig_1.pdf"):

    fig, ax = plt.subplots()

    e = 0.02

    marker_size = 60
    fs = 10

    colors = ("red", "blue", "orange", "green")
    u_0_vecs = ([[2.0], [3.0]], [[3.0], [5.2]], [[2.4], [3.6]], [[2.6], [5.6]])
    u_0_vecs = list(map(np.array, u_0_vecs))
    iter_range = 8

    for (x_0, color) in zip(u_0_vecs, colors):
        x = x_0
        s, t = x
        ax.text(s+e, t-4*e, r"$u_0$", fontsize=fs)

        for i in range(iter_range):
            s, t = x
            ax.scatter((s,), (t,), c=color, alpha=0.2, s=marker_size)
            x_new = T(x)
            s_new, t_new = x_new
            ax.plot((s, s_new), (t, t_new), marker='.',linewidth=0.5, alpha=0.5, color=color)
            x = x_new

    s_star, t_star = x_star
    ax.scatter((s_star,), (t_star,), c="k", s=marker_size * 1.2)
    ax.text(s_star-4*e, t_star+4*e, r"$u^*$", fontsize=fs)

    ax.set_xticks((2.0, 2.5, 3.0))
    ax.set_yticks((3.0, 4.0, 5.0, 6.0))
    ax.set_xlim(1.8, 3.2)
    ax.set_ylim(2.8, 6.1)

    plt.show()
    if savefig:
        fig.savefig(figname)
