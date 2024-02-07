import numpy as np
import matplotlib.pyplot as plt

xmin, xmax = 0.0000001, 2

g = lambda x: 2.125 / (1 + x**(-4))

xgrid = np.linspace(xmin, xmax, 200)

fig, ax = plt.subplots(figsize=(6.5, 6))

ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

ax.plot(xgrid, g(xgrid), 'b-', lw=2, alpha=0.6, label='$G$')
ax.plot(xgrid, xgrid, 'k-', lw=1, alpha=0.7, label='$45^o$')



ax.legend(fontsize=14)


fps = (0.01, 0.94, 1.98)
fps_labels = ('$x_\ell$', '$x_m$', '$x_h$' )
coords = ((40, 80), (40, -40), (-40, -80))

ax.plot(fps, fps, 'ro', ms=8, alpha=0.6)


for (fp, lb, coord) in zip(fps, fps_labels, coords):
    ax.annotate(lb, 
             xy=(fp, fp),
             xycoords='data',
             xytext=coord,
             textcoords='offset points',
             fontsize=16,
             arrowprops=dict(arrowstyle="->"))
    
#plt.savefig("../figures/three_fixed_points.pdf")

plt.show()


