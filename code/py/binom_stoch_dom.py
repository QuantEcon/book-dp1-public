import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


n, m, p = 10, 18, 0.5

phi = binom(n, p)
psi = binom(m, p)

x = np.arange(0, m+1)

fig, ax = plt.subplots(figsize=(9, 5.2))
lb = r"$\phi = B({}, {})$".format(n, p)
ax.plot(x, np.concatenate((phi.pmf(np.arange(0, n+1)),
                           np.zeros(m-n))), "-o", alpha=0.6, label=lb)
lb = r"$\psi = B({}, {})$".format(m, p)
ax.plot(x, psi.pmf(x), "-o", alpha=0.6, label=lb)

ax.legend(fontsize=16, frameon=False)

plt.show()

file_name = "../figures_py/binom_stoch_dom.png"
fig.savefig(file_name)
