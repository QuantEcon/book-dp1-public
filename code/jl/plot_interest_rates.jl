# Nominal interest rate from https://fred.stlouisfed.org/series/GS1
# Real interest rate from https://fred.stlouisfed.org/series/WFII10
#
# Download as CSV files
#

using DataFrames, CSV, PyPlot

df_nominal = DataFrame(CSV.File("data/GS1.csv"))
df_real = DataFrame(CSV.File("data/WFII10.csv"))

function plot_rates(df; fontsize=16, savefig=true)
    r_type = df == df_nominal ? "nominal" : "real"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df[!, 1], df[!, 2], label=r_type*" interest rate")
    ax.plot(df[!, 1], zero(df[!, 2]), c="k", ls="--")
    ax.set_xlim(df[1, 1], df[end, 1])
    ax.legend(fontsize=fontsize, frameon=false)
    plt.show()
    if savefig
        fig.savefig("./figures/plot_interest_rates_"*r_type*".pdf")
    end
end
