import numpy as np
from collections import namedtuple


# NamedTuple Model
Model = namedtuple("Model", ("α", "β"))


def create_laborer_model(α=0.3, β=0.2):
    return Model(α=α, β=β)


def laborer_update(x, model):  # update X from t to t+1
    if x == 1:
        x_ = 2 if np.random.rand() < model.α else 1
    else:
        x_ = 1 if np.random.rand() < model.β else 2
    return x_


def sim_chain(k, p, model):
    X = np.empty(k)
    X[0] = 1 if np.random.rand() < p else 2
    for t in range(0, k-1):
        X[t+1] = laborer_update(X[t], model)
    return X


def test_convergence(k=10_000_000, p=0.5):
    model = create_laborer_model()
    α, β = model
    ψ_star = (1/(α + β)) * np.array([β, α])
    X = sim_chain(k, p, model)
    ψ_e = (1/k) * np.array([sum(X == 1), sum(X == 2)])
    error = np.max(np.abs(ψ_star - ψ_e))
    approx_equal = np.allclose(ψ_star, ψ_e, rtol=0.01)
    print(f"Sup norm deviation is {error}")
    print(f"Approximate equality is {approx_equal}")
