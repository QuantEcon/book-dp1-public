"""
VFI approach to job search in the infinite-horizon IID case.

"""

include("two_period_job_search.jl")
include("s_approx.jl")

" The Bellman operator. "
function T(v, model)
    (; n, w_vals, ϕ, β, c) = model
    return [max(w / (1 - β), c + β * v'ϕ) for w in w_vals]
end

" Get a v-greedy policy. "
function get_greedy(v, model)
    (; n, w_vals, ϕ, β, c) = model
    σ = w_vals ./ (1 - β) .>= c .+ β * v'ϕ  # Boolean policy vector
    return σ
end
        
" Solve the infinite-horizon IID job search model by VFI. "
function vfi(model=default_model) 
    (; n, w_vals, ϕ, β, c) = model
    v_init = zero(model.w_vals)  
    v_star = successive_approx(v -> T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
end
    

# == Plots == #

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

# A model with default parameters
default_model = create_job_search_model()


" Plot a sequence of approximations. "
function fig_vseq(model=default_model; 
                    k=3, 
                    savefig=false, 
                    figname="./figures/iid_job_search_1.pdf",
                    fs=16)

    v = zero(model.w_vals)  
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i in 1:k
        ax.plot(model.w_vals, v, lw=3, alpha=0.6, label="iterate $i")
        v = T(v, model)
    end

    for i in 1:1000
        v = T(v, model)
    end
    ax.plot(model.w_vals, v, "k-", lw=3.0, label="iterate 1000", alpha=0.7)

    #ax.set_ylim((0, 140))
    ax.set_xlabel("wage offer", fontsize=fs)
    ax.set_ylabel("lifetime value", fontsize=fs)

    ax.legend(fontsize=fs, frameon=false)

    if savefig
        fig.savefig(figname)
    end
    plt.show()
end


" Plot the fixed point. "
function fig_vstar(model=default_model; 
                   savefig=false, fs=18,
                   figname="./figures/iid_job_search_3.pdf")

    (; n, w_vals, ϕ, β, c) = model
    v_star, σ_star = vfi(model)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(w_vals, v_star, "k-", lw=1.5, label="value function")
    cont_val = c + β * v_star'ϕ
    ax.plot(w_vals, fill(cont_val, n+1), 
            "--", 
            lw=5,
            alpha=0.5,
            label="continuation value")

    ax.plot(w_vals,
            w_vals / (1 - β),
            "--",
            lw=5,
            alpha=0.5,
            label=L"w/(1 - \beta)")

    #ax.set_ylim(0, v_star.max())
    ax.legend(frameon=false, fontsize=fs, loc="lower right")

    if savefig
        fig.savefig(figname)
    end
    plt.show()
end



