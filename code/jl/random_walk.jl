using StatsBase

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

fig, ax = plt.subplots(figsize=(9, 5.2))

n, m = 100, 12
cols = ("k-", "b-", "g-")

for i in 1:m
    s = cols[rand(1:3)]
    ax.plot(cumsum(randn(n)), s,  alpha=0.5)
end
    
ax.set_xlabel("time")
plt.show()

fig.savefig("./figures/random_walk_1.pdf")


