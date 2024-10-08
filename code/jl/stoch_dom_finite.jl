using Distributions
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

p, q = 0.75, 0.25
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
ax = axes[1]
ax.bar(1:2, (p, 1-p), label=L"\varphi")

ax = axes[2]
ax.bar(1:2, (q, 1-q), label=L"\psi")

ax.legend()
plt.show()


