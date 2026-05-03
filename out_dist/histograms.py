
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid') # Apply a pre-built style

# BITCOIN-M
bins = ["0", "1", "2-5", "6-10", ">10", ">100", ">1000"] # X-axis name for each bar
counts = [352523, 146531, 1551393, 83752, 393448, 7680, 1662]  # Height of each bar

fig, ax = plt.subplots(figsize=(8, 5)) # Create figure canvas
x = np.arange(len(bins)) # Create positions where bars will be placed
bars = ax.bar(x, counts, color='steelblue', edgecolor='white', linewidth=0.8, width=0.6) # Draw the bars

# Axis labels and titles
ax.set_xticks(x)
ax.set_xticklabels(bins, fontsize=11)
ax.set_xlabel("Out-degree", fontsize=13, labelpad=10)
ax.set_ylabel("Node count", fontsize=13, labelpad=10)
ax.set_title("Out-degree Distribution (Bitcoin-M)", fontsize=15, fontweight='bold', pad=15)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}')) # Add commas to y-axis labels
# Dont wnat border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(counts)*0.01,
            f"{count:,}",
            ha='center', va='bottom', fontsize=8.5, color='#333333'
        )

plt.tight_layout()
plt.show()


# ETH
bins = ["0", "1", "2-5", "6-10", ">10", ">100", ">1000"]
counts = [799190, 1230640, 632004, 115720, 108686, 7948, 1043]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(bins))
bars = ax.bar(x, counts, color='steelblue', edgecolor='white', linewidth=0.8, width=0.6)

ax.set_xticks(x)
ax.set_xticklabels(bins, fontsize=11)
ax.set_xlabel("Out-degree", fontsize=13, labelpad=10)
ax.set_ylabel("Node count", fontsize=13, labelpad=10)
ax.set_title("Out-degree Distribution (Eth)", fontsize=15, fontweight='bold', pad=15)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(counts)*0.01,
            f"{count:,}", # Actual num to display, formatted with comma separators
            ha='center', va='bottom', fontsize=8.5, color='#333333'
        )

plt.tight_layout()
plt.show()


# DGRAPH
bins = ["0", "1", "2-5", "6-10", ">10", ">100", ">1000"]
counts = [1509888, 949169, 1216786, 24707, 0, 0, 0]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(bins))
bars = ax.bar(x, counts, color='steelblue', edgecolor='white', linewidth=0.8, width=0.6)

ax.set_xticks(x)
ax.set_xticklabels(bins, fontsize=11)
ax.set_xlabel("Out-degree", fontsize=13, labelpad=10)
ax.set_ylabel("Node count", fontsize=13, labelpad=10)
ax.set_title("Out-degree Distribution (DGraph)", fontsize=15, fontweight='bold', pad=15)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(counts)*0.01,
            f"{count:,}",
            ha='center', va='bottom', fontsize=8.5, color='#333333'
        )

plt.tight_layout()
plt.show()