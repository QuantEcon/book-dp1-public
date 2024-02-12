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

# ## A Life Cycle Model

from initialize import *
from quantecon import LQ


# +
class LifeCycle:

    def __init__(self,
                 r=0.04, 
                 β=1/(1+r), 
                 T=45, 
                 c_bar=2, 
                 σ = 0.05, 
                 μ = 1, 
                 q = 1e6):

        print("market discount rate = ", 1/(1 + r))

        self.r = r 
        self.β = β
        self.T = T
        self.c_bar = c_bar
        self.σ = σ
        self.μ = μ 
        self.q = q

        # Formulate as an LQ problem
        Q = 1
        R = np.zeros((2, 2))
        Rf = np.zeros((2, 2))
        Rf[0, 0] = q
        A = [[1 + r, -c_bar + μ],
             [0,              1]]
        B = [[-1], [ 0]]
        C = [[σ], [0]]

        self.lq = LQ(Q, R, A, B, C, beta=self.β, T=self.T, Rf=Rf)
        
        self.x0 = (0, 1)
        self.xp, self.up, self.wp = \
            self.lq.compute_sequence(self.x0, random_state=1234)

        # Convert back to assets, consumption and income
        self.assets = self.xp[0, :]                     # a_t
        self.c = self.up.flatten() + self.c_bar         # c_t
        self.income = self.σ * self.wp[0, 1:] + self.μ  # y_t

    def plot(self, ax0, ax1):

        p_args = {'lw': 2, 'alpha': 0.7}

        ax0.plot(list(range(1, T+1)), 
                     self.income,  
                     label="non-financial income",
                     **p_args)
        ax0.plot(list(range(T)), 
                     self.c, 
                     label="consumption", 
                     **p_args)
        ax0.plot(list(range(T)), 
                     self.μ * np.ones(T), 
                     'k--', 
                     lw=0.75)

        ax1.plot(list(range(1, T+1)), 
                     np.cumsum(self.income - self.μ),
                     label="cumulative unanticipated income", 
                     **p_args)
        ax1.plot(list(range(T+1)), 
                     self.assets, 
                     label="assets", 
                     **p_args)
        ax1.plot(list(range(T)), 
                     np.zeros(T), 
                     'k--', 
                     lw=0.5)

        yl, yu = μ-3*σ, μ+3*σ
        ax0.set_ylim((yl, yu))
        ax0.set_yticks((yl, μ, yu))

        yl, yu = -9*σ, 9*σ
        ax1.set_ylim((yl, yu))
        ax1.set_yticks((yl, 0.0, yu))

        for ax in (ax0, ax1):
            ax.set_xlabel('time')
            ax.legend(fontsize=12, loc='upper left', frameon=False)




lc1 = LifeCycle()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.25))
lc1.plot(axes[0], axes[1])

plt.savefig("./figures/lqcontrol_1.pdf")
plt.show()



# +
lc2 = LifeCycle(β=0.962)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.25))
lc2.plot(axes[0], axes[1])

plt.savefig("./figures/lqcontrol_2.pdf")
plt.show()



# +
lc3 = LifeCycle(β=0.96)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.25))
lc3.plot(axes[0], axes[1])

plt.savefig("./figures/lqcontrol_3.pdf")
plt.show()


# -




