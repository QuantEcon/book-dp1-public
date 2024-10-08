"""
Job search with Markov wage draws and quantile preferences.

"""
from quantecon import tauchen, MarkovChain
import numpy as np
from quantile_function import R
import matplotlib.pyplot as plt

"Creates an instance of the job search model."
def create_markov_js_model(
        n=100,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.98,      # discount factor
        c=1.0,       # unemployment compensation
        τ=0.5        # quantile parameter
    ):
    mc = tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P
    return (n, w_vals, P, β, c, τ)

"""
The policy operator 

    (T_σ v)(w) = σ(w) (w / (1-β)) + (1 - σ(w))(c + β (R_τ v)(w))

"""
def T_σ(v, σ, model):
    n, w_vals, P, β, c, τ = model
    h = [x + c for x in β * R(τ, v, P)]
    e = w_vals / (1 - β)
    return σ * e + (1 - σ) * h


def get_greedy(v, model):
    """Get a v-greedy policy."""
    n, w_vals, P, β, c, τ = model
    σ = w_vals / (1 - β) >= c + β * R(τ, v, P)
    return σ


def optimistic_policy_iteration(model, tolerance=1e-5, m=100):
    """Optimistic policy iteration routine."""
    n, w_vals, P, β, c, τ = model
    v = np.ones(n)
    error = tolerance + 1
    while error > tolerance:
        last_v = v
        σ = get_greedy(v, model)
        for i in range(m):
            v = T_σ(v, σ, model)
        
        error = max(np.abs(v - last_v))
        print(f"OPI current error = {error}")
    
    return v, get_greedy(v, model)


# == Plots == #

def plot_main(tau_vals=(0.1, 0.25, 0.5, 0.6, 0.7, 0.8), 
                     savefig=False, 
                     figname="../figures_py/quantile_js.png"):

    w_star_vals = np.zeros(len(tau_vals))

    for (τ_idx, τ) in enumerate(tau_vals):
        model = create_markov_js_model(τ=τ)
        n, w_vals, P, β, c, τ = model
        v_star, σ_star = optimistic_policy_iteration(model)
        for i in range(n):
            if σ_star[i] > 0:
                w_star_vals[τ_idx] = w_vals[i]
                break

    model = create_markov_js_model()
    n, w_vals, P, β, c, τ = model
    mc = MarkovChain(P)
    s = mc.stationary_distributions[0]

    fig, ax = plt.subplots()
    ax.plot(tau_vals, w_star_vals, "k--", lw=2, alpha=0.7, label="reservation wage")
    ax.barh(w_vals, 32 * s, alpha=0.05, align="center")
    ax.legend(frameon=False, loc="upper center")
    ax.set_xlabel("quantile")
    ax.set_ylabel("wages")

    plt.show()
    if savefig:
        fig.savefig(figname)

plot_main(savefig=True)
