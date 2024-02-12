using PyPlot, LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

fs = 20

xmin, xmax = 0., 1.0

g(x) = 0.2 + 0.6 * x^(1.2)

xgrid = LinRange(xmin, xmax, 200)

fig, ax = plt.subplots(figsize=(8.0, 6))
for spine in ["left", "bottom"]
    ax.spines[spine].set_position("zero")
end
for spine in ["right", "top"]
    ax.spines[spine].set_color("none")
end

ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

ax.plot(xgrid, g.(xgrid), "b-", lw=2, alpha=0.6, label=L"T")
ax.plot(xgrid, xgrid, "k-", lw=1, alpha=0.7, label=L"45^o")

ax.legend(frameon=false, fontsize=fs)

fp = (0.4,)
fps_label = L"\bar v"
coords = (40, -20)

ax.plot(fp, fp, "ro", ms=8, alpha=0.6)

ax.set_xticks([0, 1])
ax.set_xticklabels([L"0", L"1"], fontsize=fs)
ax.set_yticks([])

ax.set_xlabel(L"V", fontsize=fs)

ax.annotate(fps_label, 
         xy=(fp[1], fp[1]),
         xycoords="data",
         xytext=coords,
         textcoords="offset points",
         fontsize=fs,
         arrowprops=Dict("arrowstyle"=>"->"))
    
plt.savefig("./figures/up_down_stable.pdf")

plt.show()


