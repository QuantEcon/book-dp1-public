using Distributions

n, m, p = 10, 18, 0.5
ϕ = Binomial(n, p)
ψ = Binomial(m, p)


x = 0:m


# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

fig, ax = plt.subplots(figsize=(9, 5.2))
lb = latexstring("\\phi = B($n, $p)")
ax.plot(x, vcat(pdf(ϕ), zeros(m-n)), "-o", alpha=0.6, label=lb)
lb = latexstring("\\psi = B($m, $p)")
ax.plot(x, pdf(ψ), "-o", alpha=0.6, label=lb)

ax.legend(fontsize=16, frameon=false)

if false
    fig.savefig("./figures/binom_stoch_dom.pdf")
end

plt.show()

