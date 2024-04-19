using PyPlot, LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

fs = 18

xmin=-0.5
xmax=2.0

xgrid = LinRange(xmin, xmax, 1000)

a1, b1 = 0.15, 0.5    # first T_σ
a2, b2 = 0.5, 0.4     # second T_σ
a3, b3 = 0.75, 0.2     # third T_σ

v1 = b1/(1-a1)
v2 = b2/(1-a2)
v3 = b3/(1-a3)

T1 = a1 * xgrid .+ b1
T2 = a2 * xgrid .+ b2
T3 = a3 * xgrid .+ b3
T = max.(T1, T2, T3)

fig, ax = plt.subplots()
for spine in ["left", "bottom"]
    ax.spines[spine].set_position("zero")
end
for spine in ["right", "top"]
    ax.spines[spine].set_color("none")
end

ax.plot(xgrid, T1, "k-", lw=1)
ax.plot(xgrid, T2, "k-", lw=1)
ax.plot(xgrid, T3, "k-", lw=1)

ax.plot(xgrid, T, lw=6, alpha=0.3, color="blue", label=L"T = \bigvee_{\sigma \in \Sigma} T_\sigma")


ax.text(2.1, 0.6, L"T_{\sigma'}", fontsize=fs)
ax.text(2.1, 1.4, L"T_{\sigma''}", fontsize=fs)
ax.text(2.1, 1.9, L"T_{\sigma'''}", fontsize=fs)

ax.legend(frameon=false, loc="upper center", fontsize=fs)


ax.set_xlim(xmin, xmax+0.5)
ax.set_ylim(-0.2, 2)
ax.text(2.4, -0.15, L"v", fontsize=22)

ax.set_xticks([])
ax.set_yticks([])

plt.show()

file_name = "../figures/bellman_envelope.pdf"
fig.savefig(file_name)

