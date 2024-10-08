import numpy as np
import matplotlib.pyplot as plt
import time

from ez_model import create_ez_model
from ez_dp_code import optimistic_policy_iteration2, optimistic_policy_iteration

n_vals = np.arange(20, 101, 10)
β_vals = [0.96, 0.98]
gains = np.zeros((len(β_vals), len(n_vals)))

for β_i, β in enumerate(β_vals):
    for n_i, n in enumerate(n_vals):
        model = create_ez_model(n=n, β=β)
        v_init = np.ones((len(model.w_grid), len(model.e_grid)))
        
        in_time = time.time()
        optimistic_policy_iteration(v_init, model)
        unmod_time = time.time() - in_time

        h_init = np.ones(len(model.w_grid))
        in_time = time.time()
        optimistic_policy_iteration2(h_init, model)
        mod_time = time.time() - in_time
        
        gains[β_i, n_i] = unmod_time / mod_time

def plot_function(savefig=False, figname="../figures_py/ez_rel_timing.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    global gains, n_vals, β_vals
    for β_i, β in enumerate(β_vals):
        label = f"speed gain with $\\beta$ = {β}"
        ax.plot(n_vals, gains[β_i, :], "-o", label=label)

    ax.legend(loc="lower right", fontsize=16)
    ax.set_xticks(n_vals)
    ax.set_xlabel("size of $\\mathsf{E}$", fontsize=16)
    ax.set_ylabel("Speed Gain", fontsize=16)
    if savefig:
        plt.savefig(figname)
    plt.show()

plot_function(True)
