
"""
Optimal policy plot

"""

include("ez_model.jl")
include("ez_dp_code.jl")
include("ez_plot_functions.jl")


model = create_ez_model()
(; α, β, γ, θ, φ, e_grid, w_grid) = model


println("Solving modified model.")
h_init = ones(length(model.w_grid))
@time h_star, _ = optimistic_policy_iteration(h_init, model)

σ_star_mod = G_max(h_star, model)
plot_policy(σ_star_mod, model, title="optimal savings", savefig=true)
