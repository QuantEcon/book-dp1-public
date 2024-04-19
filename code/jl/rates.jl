
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16


qs = (1, 2)
labels = "linear", "quadratic"
x0 = 0.9
N = 26
β = 0.9

fig, ax = plt.subplots()

for (q, label) in zip(qs, labels)
    current_x = x0
    x = []
    for t in 1:N
        push!(x, current_x)
        current_x = β * current_x^q
    end
    ax.plot(x, "o-", label=label, alpha=0.6)
end

ax.legend(fontsize=16)

plt.savefig("../figures/rates_1.pdf")
plt.show()
