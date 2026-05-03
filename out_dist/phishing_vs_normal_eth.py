import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

metrics = ["Mean out-degree", "Mean in-degree", "Median out-degree", "Median in-degree"]
normal   = [4.49, 4.49, 1.0, 0.0]
phishing = [19.35, 31.08, 2.0, 9.0]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars_n = ax.bar(x - width/2, normal,   width, label='Normal',   color='steelblue', edgecolor='white', linewidth=0.8)
bars_p = ax.bar(x + width/2, phishing, width, label='Phishing', color='tomato',    edgecolor='white', linewidth=0.8)

for bar, val in zip(bars_n, normal):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8.5, color='#333333')

for bar, val in zip(bars_p, phishing):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8.5, color='#333333')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Degree", fontsize=13, labelpad=10)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('out_dist/phishing_vs_normal_eth.pdf', bbox_inches='tight')
plt.show()
