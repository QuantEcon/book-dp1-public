"""

Timing figure

"""

include("ez_model.jl")
include("ez_dp_code.jl")
include("ez_plot_functions.jl")


n_vals = [i * 10 for i in 2:10]
β_vals = [0.96, 0.98]
gains = zeros(length(β_vals), length(n_vals))

for (β_i, β) in enumerate(β_vals)
    for (n_i, n) in enumerate(n_vals)

        model = create_ez_model(n=n, β=β)
        (; α, β, γ, θ, φ, e_grid, w_grid) = model

        println("Solving unmodified model at n = $n.")
        v_init = ones(length(model.w_grid), length(model.e_grid))
        unmod_time = @elapsed v_star, σ_star = 
            optimistic_policy_iteration(v_init, model)

        println("Solving modified model at n = $n.")
        h_init = ones(length(model.w_grid))
        mod_time = @elapsed h_star, _ = optimistic_policy_iteration(h_init, model)

        gains[β_i, n_i] = unmod_time / mod_time
    end
end


using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

fig, ax = plt.subplots(figsize=(9,5))
b = β_vals[1]
lb = "speed gain with " * L"\beta" * " = $b"
ax.plot(n_vals, gains[1, :], "-o", label=lb)
b = β_vals[2]
lb = "speed gain with " * L"\beta" * " = $b"
ax.plot(n_vals, gains[2, :], "-o", label=lb)
ax.legend(loc="lower right", fontsize=fontsize)
ax.set_xticks(n_vals)
ax.set_xlabel("size of " * L"\mathsf E", fontsize=fontsize)
plt.savefig("rel_timing.pdf")
plt.show()

