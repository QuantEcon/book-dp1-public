using Random, Distributions

"""
Generate a path for inventory starting at b, up to time T.

Return the path as a function X(t) constructed from (J_k) and (Y_k).
"""
function sim_path(; T=10, seed=123, λ=0.5, α=0.7, b=10)

    J, Y = 0.0, b
    J_vals, Y_vals = [J], [Y]
    Random.seed!(seed)
    φ = Exponential(1/λ)     # Wait times are exponential
    G = Geometric(α)         # Orders are geometric

    while true
        W = rand(φ)  
        J += W
        push!(J_vals, J)
        if Y == 0
            Y = b
        else
            U = rand(G) + 1   # Geometric on 1, 2,...
            Y = Y - min(Y, U)
        end
        push!(Y_vals, Y)
        if J > T
            break
        end
    end
    
    function X(t)
        k = searchsortedlast(J_vals, t)
        return Y_vals[k+1]
    end

    return X
end



T = 50
X = sim_path(T=T)

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

grid = LinRange(0, T - 0.001, 100)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.step(grid, [X(t) for t in grid], label=L"X_t", alpha=0.7)

ax.set(xticks=(0, 10, 20, 30, 40, 50))

ax.set_xlabel("time", fontsize=12) 
ax.set_ylabel("inventory", fontsize=12)
ax.legend(fontsize=12)

plt.savefig("../figures/inventory_cont_time_1.pdf")
plt.show()



