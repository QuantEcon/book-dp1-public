using QuantEcon 

ρ, b, ν = 0.9, 0.0, 1.0
μ_x = b/(1-ρ)
σ_x = sqrt(ν^2 / (1-ρ^2))

n = 15
mc = tauchen(n, ρ, ν)
approx_sd = stationary_distributions(mc)[1]

function psi_star(y)
    c = 1/(sqrt(2 * pi) * σ_x)
    return c * exp(-(y - μ_x)^2 / (2 * σ_x^2))
end

# == Plots == #
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=14

fig, ax = plt.subplots()
ax.bar(mc.state_values, approx_sd, 
       fill=true, width=0.6, alpha=0.6, 
       label="approximation")

x_grid = LinRange(minimum(mc.state_values) - 2, 
                  maximum(mc.state_values) + 2, 100)
ax.plot(x_grid, psi_star.(x_grid), 
        "-k", lw=2, alpha=0.6, label=L"\psi^*")
ax.legend(fontsize=fontsize)
if false
    plt.savefig("./figures/tauchen_1.pdf")
end
plt.show()


