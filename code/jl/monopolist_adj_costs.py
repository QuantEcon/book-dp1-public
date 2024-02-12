# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Monopolist with adjustment costs

# +
from initialize import *

import control

# +
# == Model parameters == #
a0 = 5
a1 = 0.5
σ = 0.15
ρ = 0.9
γ = 10
β = 0.95
c = 2
T = 120

# == Useful constants == #
m0 = (a0-c)/(2 * a1)
m1 = 1/(2 * a1)

# == Formulate LQ problem == #
Q = γ
R = [[ a1, -a1,  0],
     [-a1,  a1,  0],
     [  0,   0,  0]]
A = [[ρ, 0, m0 * (1 - ρ)],
     [0, 1,            0],
     [0, 0,            1]]

B = [[0],
     [1],
     [0]]
C = [[m1 * σ],
     [     0],
     [     0]]
# -

A, R = np.array(A), np.array(R)
np.linalg.matrix_rank(control.ctrb(A.T, R.T))

# +
lq = qe.LQ(Q, R, A, B, C=C, beta=β)

# == Simulate state / control paths == #
x0 = (m0, 2, 1)
xp, up, wp = lq.compute_sequence(x0, ts_length=150, random_state=123)
q_bar = xp[0, :]
q = xp[1, :]

# == Plot simulation results == #
fig, ax = plt.subplots(figsize=(10, 6.5))

# == Some fancy plotting stuff -- simplify if you prefer == #
bbox = (0., 1.01, 1., .101)
legend_args = {'bbox_to_anchor': bbox, 'loc': 3, 'mode': 'expand'}
p_args = {'lw': 2, 'alpha': 0.6}

time = range(len(q))
ax.set(xlabel='Time', xlim=(0, max(time)))
ax.plot(time, q_bar, 'k-', lw=2, alpha=0.6, label=r'$\bar q_t$')
ax.plot(time, q, 'b-', lw=2, alpha=0.6, label='$q_t$')
ax.legend(ncol=2, **legend_args)
s = f'dynamics with $\gamma = {γ}$'
ax.text(max(time) * 0.6, 1 * q_bar.max(), s, fontsize=14)
plt.show()
# -


