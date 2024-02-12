"""
Continuation value function approach to job search in the IID case.

"""

include("iid_job_search.jl")
include("s_approx.jl")

function g(h, model)
    (; n, w_vals, ϕ, β, c) = model
    return c + β * max.(w_vals / (1 - β), h)'ϕ
end

function compute_hstar_wstar(model, h_init=0.0)
    (; n, w_vals, ϕ, β, c) = model
    h_star = successive_approx(h -> g(h, model), h_init)
    return h_star, (1 - β) * h_star
end


# == Plots == #

" Plot the function g. "
function fig_g(model=default_model; 
               savefig=false, fs=18,
               figname="./figures/iid_job_search_g.pdf")

    (; n, w_vals, ϕ, β, c) = model
    h_grid = collect(LinRange(600, max(c, n) / (1 - β), 100))
    g_vals = [g(h, model) for h in h_grid]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(h_grid, g_vals, lw=2.0, label=L"g")
    ax.plot(h_grid, h_grid, "k--", lw=1.0, label="45")

    ax.legend(frameon=false, fontsize=fs, loc="lower right")

    h_star, w_star = compute_hstar_wstar(model)
    ax.plot(h_star, h_star, "go", ms=10, alpha=0.6)

    ax.annotate(L"$h^*$", 
             xy=(h_star, h_star),
             xycoords="data",
             xytext=(40, -40),
             textcoords="offset points",
             fontsize=fs)

    if savefig
        fig.savefig(figname)
    end

    plt.show()

end


" Plot the two ordered instances of function g. "
function fig_tg(betas=[0.95, 0.96]; 
               savefig=false, fs=18,
               figname="./figures/iid_job_search_tg.pdf")

    h_grid = collect(LinRange(600, 1200, 100))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(h_grid, h_grid, "k--", lw=1.0, label="45")

    for (i, β) in enumerate(betas)
        model = create_job_search_model(β=β)
        (; n, w_vals, ϕ, β, c) = model
        b = maximum(betas)
        g_vals = [g(h, model) for h in h_grid]

        lb = latexstring("g_$i \\; (\\beta_$i = $β)")
        ax.plot(h_grid, g_vals, lw=2.0, label=lb)

        ax.legend(frameon=false, fontsize=fs, loc="lower right")

        h_star, w_star = compute_hstar_wstar(model)
        ax.plot(h_star, h_star, "go", ms=10, alpha=0.6)

        lb = latexstring("h^*_$i")
        ax.annotate(lb, 
                 xy=(h_star, h_star),
                 xycoords="data",
                 xytext=(40, -40),
                 textcoords="offset points",
                 fontsize=fs)
    end

    if savefig
        fig.savefig(figname)
    end

    plt.show()

end


" Plot continuation value and the fixed point. "
function fig_cv(model=default_model;
                fs=18,
                savefig=false, 
                figname="./figures/iid_job_search_4.pdf")

    (; n, w_vals, ϕ, β, c) = model
    h_star, w_star = compute_hstar_wstar(model)
    vhat = max.(w_vals / (1 - β), h_star)

    fig, ax = plt.subplots()
    ax.plot(w_vals, vhat, "k-", lw=2.0, label="value function")
    ax.legend(fontsize=fs)
    ax.set_ylim(0, maximum(vhat))

    plt.show()
    if savefig
        fig.savefig(figname)
    end
end


" Plot the fixed point as a function of β. "
function fig_bf(betas=LinRange(0.9, 0.99, 20); 
                savefig=false, 
                fs=16,
                figname="./figures/iid_job_search_bf.pdf")

    h_vals = similar(betas)
    for (i, β) in enumerate(betas)
        model = create_job_search_model(β=β)
        h, w = compute_hstar_wstar(model)
        h_vals[i] = h
    end

    fig, ax = plt.subplots()
    ax.plot(betas, h_vals, lw=2.0, alpha=0.7, label=L"h^*(\beta)")
    ax.legend(frameon=false, fontsize=fs)
    ax.set_xlabel(L"\beta", fontsize=fs)
    ax.set_ylabel("continuation value", fontsize=fs)

    if savefig
        fig.savefig(figname)
    end

    plt.show()

end
