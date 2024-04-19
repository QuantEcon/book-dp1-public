using PyPlot, LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

fs = 18

xmin, xmax = 0.0000001, 2.0

g(u) = 2.125 / (1 + u^(-4))

xgrid = LinRange(xmin, xmax, 200)

fig, ax = plt.subplots(figsize=(6.5, 6))

ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

ax.plot(xgrid, g.(xgrid), "b-", lw=2, alpha=0.6, label=L"T")
ax.plot(xgrid, xgrid, "k-", lw=1, alpha=0.7, label=L"45^o")

ax.legend(fontsize=14)

fps = (0.01, 0.94, 1.98)
fps_labels = (L"u_\ell", L"u_m", L"u_h" )
coords = ((40, 80), (40, -40), (-40, -80))

ax.plot(fps, fps, "ro", ms=8, alpha=0.6)

for (fp, lb, coord) in zip(fps, fps_labels, coords)
    ax.annotate(lb, 
             xy=(fp, fp),
             xycoords="data",
             xytext=coord,
             textcoords="offset points",
             fontsize=16,
             arrowprops=Dict("arrowstyle"=>"->"))
end
    
plt.savefig("../figures/three_fixed_points.pdf")

plt.show()


