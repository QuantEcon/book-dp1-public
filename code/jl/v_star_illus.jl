using PyPlot, LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

fs = 18

xmin=0.01
xmax=2.0

xgrid = LinRange(xmin, xmax, 1000)


v1 = xgrid.^0.7
v2 = xgrid.^0.1 .+ .05
v = max.(v1, v2)

fig, ax = plt.subplots()

for spine in ["left", "bottom"]
    ax.spines[spine].set_position("zero")
end
for spine in ["right", "top"]
    ax.spines[spine].set_color("none")
end

ax.plot(xgrid, v1, "k-", lw=1)
ax.plot(xgrid, v2, "k-", lw=1)

ax.plot(xgrid, v, lw=6, alpha=0.3, color="blue", 
        label=L"v^* = \bigvee_{\sigma \in \Sigma} v_\sigma")

ax.text(2.1, 1.1, L"v_{\sigma'}", fontsize=fs)
ax.text(2.1, 1.6, L"v_{\sigma''}", fontsize=fs)

ax.text(1.2, 0.3, L"\Sigma = \{\sigma', \sigma''\}", fontsize=fs)

ax.legend(frameon=false, loc="upper left", fontsize=fs)

ax.set_xlim(xmin, xmax+0.5)
ax.set_ylim(0.0, 2)
ax.text(2.4, -0.15, L"x", fontsize=20)

ax.set_xticks([])
ax.set_yticks([])

plt.show()

file_name = "./figures/v_star_illus.pdf"
fig.savefig(file_name)

