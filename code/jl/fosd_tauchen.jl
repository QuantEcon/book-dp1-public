using Distributions
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

n = 25
ν = 1.0
a = 0.5

mc = tauchen(n, a, ν)
i, j = 8, 12


fig, axes = plt.subplots(2, 1, figsize=(10, 5.2))
fontsize=16
ax = axes[1]

ax.plot(mc.state_values, mc.p[i, :], "b-o", alpha=0.4, lw=2, label=L"\varphi")
ax.plot(mc.state_values, mc.p[j, :], "g-o", alpha=0.4, lw=2, label=L"\psi")
ax.legend(frameon=false, fontsize=fontsize)

ax = axes[2]
F = [sum(mc.p[i, k:end]) for k in 1:n]
G = [sum(mc.p[j, k:end]) for k in 1:n]
ax.plot(mc.state_values, F, "b-o", alpha=0.4, lw=2, label=L"G^\varphi")
ax.plot(mc.state_values, G, "g-o", alpha=0.4, lw=2, label=L"G^\psi")
ax.legend(frameon=false, fontsize=fontsize)

plt.show()

fig.savefig("../figures/fosd_tauchen_1.pdf")

