from pandas import read_csv
from pathlib import Path

import matplotlib.pyplot as plt


temperatures = (10, 300, 1000)

fig, axes = plt.subplots(nrows=len(temperatures), ncols=2, figsize=(12, 4),
                         sharex='col', sharey='row')

for ax_row, temperature in zip(axes, temperatures):
    print(temperature)
    for ax, run in zip(ax_row, ('eq_', '')):
        data = read_csv(f'md_{run}{temperature}K.log', sep=r'\s+')
        ax.plot(data['Time[ps]'], data['T[K]'])
    ax_row[0].set_title("Equilibration")
    ax_row[0].set_ylabel("Temperature / K")
    ax_row[1].set_title(f"{temperature}K run")

for ax in axes[-1]:
    ax.set_xlabel("Time / ps")

fig.tight_layout()
fig.savefig("temperature_logs.pdf")
