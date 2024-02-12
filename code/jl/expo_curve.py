# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("text", usetex=True) # allow tex rendering
fontsize=16
from scipy.linalg import expm, eigvals

# %matplotlib inline

# +
A = ((-2, -0.4, 0),
     (-1.4, -1, 2.2),
     (0, -2, -0.6))

A = np.array(A)
# -

ev = eigvals(A)

np.imag(ev)

np.max(np.real(ev))

h = 0.01
s0 = 0.01 * np.array((1, 1, 1))

x, y, z = [], [], []
s = s0
for i in range(6000):
    s = expm(h * A) @ s
    a, b, c = s
    x.append(a)
    y.append(b)
    z.append(c)

# +
ax = plt.figure().add_subplot(projection='3d')

ax.plot(x, y, z, label='$t \mapsto \mathrm{e}^{t A} u_0$')
ax.legend()

ax.view_init(23, -132)

ax.set_xticks((-0.002, 0.002, 0.006, 0.01))
ax.set_yticks((-0.002, 0.002, 0.006, 0.01))
ax.set_zticks((-0.002, 0.002, 0.006, 0.01))

ax.set_ylim((-0.002, 0.014))

ax.text(s0[0]-0.001, s0[1]+0.002, s0[2], "$u_0$", fontsize=14)
ax.scatter(s0[0], s0[1], s0[2], color='k')

plt.savefig("./figures/expo_curve_1.pdf")
plt.show()
# -




