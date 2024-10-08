import numpy as np
import matplotlib.pyplot as plt

fontsize = 16

fig, ax = plt.subplots(figsize=(9, 5.2))

n, m = 100, 12
cols = ["k-", "b-", "g-"]

for _ in range(m):
    s = np.random.choice(cols)
    ax.plot(np.cumsum(np.random.randn(n)), s, alpha=0.5)

ax.set_xlabel("time")
plt.show()

file_name = "../figures_py/random_walk_1.png"
fig.savefig(file_name)
