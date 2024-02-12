
using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

function F(w; r=1, β=0.5, θ=5)
    return (r + β * w^(1/θ))^θ
end

w_grid = LinRange(0.1, 2.0, 200)

fig, axes = plt.subplots(2, 2)

θ_vals = -2, -0.5, 0.5, 2

for (θ, ax) in zip(θ_vals, axes)

    f(w) = F(w; θ=θ)
    ax.plot(w_grid, w_grid, "k--", alpha=0.6, label=L"45")
    ax.plot(w_grid, f.(w_grid), label=L"U")
    ax.legend()
    title = latexstring("\$\\theta = $θ\$")
    ax.set_title(title)
end

plt.show()


