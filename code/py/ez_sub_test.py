"""
Quick test and plots

"""

import numpy as np
import time

from ez_model import create_ez_model, G_max
from ez_dp_code import optimistic_policy_iteration2, optimistic_policy_iteration
from ez_plot_functions import plot_policy



model = create_ez_model()

print("Solving unmodified model.")
v_init = np.ones((len(model.w_grid), len(model.e_grid)))
in_time = time.time()
v_star, σ_star = optimistic_policy_iteration(v_init, model)
out_time = time.time()
print("Time to solve unmodified model: ", out_time - in_time)
print("Solving modified model.")

h_init = np.ones(len(model.w_grid))
in_time = time.time()
h_star, _ = optimistic_policy_iteration2(h_init, model)
out_time = time.time()
print("Time to solve modified model: ", out_time - in_time)

σ_star_mod = G_max(h_star, model)

plot_policy(σ_star, model, title="original")
plot_policy(σ_star_mod, model, title="transformed")
