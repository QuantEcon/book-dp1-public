# Nominal interest rate from https://fred.stlouisfed.org/series/GS1
# Real interest rate from https://fred.stlouisfed.org/series/WFII10
#
# Download as CSV files
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_nominal = pd.read_csv("./data/GS1.csv")
df_real = pd.read_csv("./data/WFII10.csv")

def plot_rates(df, fontsize=16, savefig=False):
    r_type = 'nominal' if df.equals(df_nominal) else 'real'
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'{r_type} interest rate')
    ax.plot(df.iloc[:, 0], np.zeros(df.iloc[:, 1].size), c='k', ls='--')
    ax.set_xlim(df.iloc[0, 0], df.iloc[-1, 0])
    ax.legend(fontsize=fontsize, frameon=False)
    plt.show()
    if savefig:
        fig.savefig(f'./figures/plot_interest_rates_{r_type}.pdf')