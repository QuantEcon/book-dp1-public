import matplotlib.pyplot as plt

fontsize = 16

qs = [1, 2]
labels = ["linear", "quadratic"]
x0 = 0.9
N = 26
beta = 0.9

fig, ax = plt.subplots()

for q, label in zip(qs, labels):
    current_x = x0
    x = []
    for t in range(1, N + 1):
        x.append(current_x)
        current_x = beta * current_x**q
    ax.plot(x, "o-", label=label, alpha=0.6)

ax.legend(fontsize=16)

file_name = "../figures_py/rates_1.png"
plt.savefig(file_name)
plt.show()
