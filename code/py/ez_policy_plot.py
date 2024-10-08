import numpy as np

from ez_model import create_ez_model, G_max
from ez_dp_code import optimistic_policy_iteration2
from ez_plot_functions import plot_policy

model = create_ez_model()

h_init = np.ones(len(model.w_grid))
h_star, _ = optimistic_policy_iteration2(h_init, model)

σ_star_mod = G_max(h_star, model)
plot_policy(σ_star_mod, model, title="optimal savings", savefig=True)
