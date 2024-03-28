include("linear_iter.jl")
using PyPlot

PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

fig, ax = plt.subplots()

e = 0.02

marker_size = 60
fs=14

colors = ("red", "blue", "orange", "green")
u_0_vecs = ([2.0; 3.0], [3.0; 5.2], [2.4; 3.6], [2.6, 5.6])
iter_range = 8

for (u_0, color) in zip(u_0_vecs, colors)
    u = u_0
    s, t = u
    ax.text(s+e, t-4e, L"u_0", fontsize=fs)

    for i in 1:iter_range
        s, t = u
        ax.scatter((s,), (t,), c=color, alpha=0.3, s=marker_size)
        u_new = T(u)
        s_new, t_new = u_new
        ax.plot((s, s_new), (t, t_new), lw=0.5, alpha=0.5, c=color)
        u = u_new
    end
end

s_star, t_star = u_star
ax.scatter((s_star,), (t_star,), c="k", s=marker_size * 1.2)
ax.text(s_star-4e, t_star+4e, L"u^*", fontsize=fs)

ax.set_xticks((2.0, 2.5, 3.0))
ax.set_yticks((3.0, 4.0, 5.0, 6.0))
ax.set_xlim(1.8, 3.2)
ax.set_ylim(2.8, 6.1)

plt.show()
fig.savefig("../figures/linear_iter_fig_1.pdf")

