from quantecon import MarkovChain
import numpy as np

P = np.array([
    [0.1, 0.9],
    [0.0, 1.0]
])

mc = MarkovChain(P)
print(mc.is_irreducible)
