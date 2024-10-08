import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

p, q = 0.75, 0.25
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))

# First subplot
ax = axes[0]
ax.bar([1, 2], [p, 1-p], label=r"$\phi$")
ax.set_xticks([1, 2])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Second subplot
ax = axes[1]
ax.bar([1, 2], [q, 1-q], label=r"$\psi$")
ax.set_xticks([1, 2])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Add legends
for ax in axes:
    ax.legend()

plt.show()
