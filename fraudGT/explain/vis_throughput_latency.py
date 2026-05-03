import matplotlib.pyplot as plt
import numpy as np


DATA = {
    "Elliptic": [
        ("MLP",                813774.8, 2.5,  0.2),
        ("GINE",               377172.2, 5.4,  0.2),
        ("Multi-GINE",         335384.3, 6.1,  0.2),
        ("GATE",               307192.7, 6.7,  0.3),
        ("Multi-GINE+EU",      302148.0, 6.8,  0.2),
        ("PNA",                232601.4, 8.8,  0.5),
        ("Multi-PNA",          220928.7, 9.3,  1.1),
        ("Multi-PNA+EU",       198155.5, 10.3, 0.3),
        ("PE-FraudGT",         181835.3, 11.3, 0.3),
        ("SparseNodeGT",       173576.3, 11.8, 0.7),
        ("Multi-SparseNodeGT", 147200.7, 13.9, 0.3),
    ],
    "ETH": [
        ("MLP",                212491.5, 9.6,  0.4),
        ("GINE",               193659.6, 10.6, 0.1),
        ("PE-FraudGT",         174778.1, 11.7, 0.4),
        ("GATE",               162742.1, 12.6, 0.2),
        ("PNA",                146832.1, 13.9, 0.3),
        ("SparseNodeGT",       114526.3, 17.9, 0.6),
        ("Multi-PNA",           69089.3, 29.6, 4.5),
        ("Multi-GINE",          66517.2, 30.8, 0.7),
        ("Multi-PNA+EU",        51845.3, 39.5, 1.2),
        ("Multi-GINE+EU",       51137.1, 40.0, 1.3),
        ("Multi-SparseNodeGT",  50195.5, 40.8, 1.1),
    ],
    "DGraph": [
        ("MLP",                           227619.7, 9.0,  0.3),
        ("PE-FraudGT",                    180597.9, 11.3, 0.2),
        ("Multi-SparseNodeGT-dropout0.5", 164957.9, 12.4, 0.4),
        ("GINE",                          164703.5, 12.4, 0.5),
        ("Multi-GINE+EU",                 152000.2, 13.5, 0.3),
        ("GATE",                          150811.2, 13.6, 0.5),
        ("PNA",                           138433.5, 14.8, 0.6),
        ("Multi-GINE",                    112391.0, 18.2, 0.4),
        ("Multi-PNA",                     110442.0, 18.5, 0.6),
        ("Multi-PNA+EU",                   99784.4, 20.5, 1.1),
        ("SparseNodeGT",                   97590.8, 21.0, 1.2),
        ("Multi-SparseNodeGT-dropout0.35", 93086.0, 22.0, 1.5),
        ("Multi-SparseNodeGT",             87493.9, 23.4, 1.2),
    ],
    "Bitcoin-M": [
        ("PE-FraudGT",         112117.6, 18.3,  0.5),
        ("PNA",                 92521.9, 22.1,  1.8),
        ("MLP",                 89564.9, 22.9,  0.6),
        ("SparseNodeGT",        87487.5, 23.4,  0.8),
        ("GINE",                86226.0, 23.8,  0.6),
        ("GATE",                80107.7, 25.6,  1.3),
        ("Multi-PNA",           21997.7, 93.1,  2.0),
        ("Multi-SparseNodeGT",  20207.2, 101.3, 7.7),
        ("Multi-GINE+EU",       17419.5, 117.6, 2.3),
        ("Multi-GINE",          15288.5, 134.0, 13.0),
        ("Multi-PNA+EU",        15127.2, 135.4, 16.5),
    ],
}

# Fixed global order — all unique model names across datasets, sorted alphabetically
ALL_MODELS = sorted({name for rows in DATA.values() for name, *_ in rows})
COLOR_MAP = {name: plt.cm.tab20(i / len(ALL_MODELS)) for i, name in enumerate(ALL_MODELS)}


def plot_throughput_latency(dataset):
    rows = {name: (tp, lat, sd) for name, tp, lat, sd in DATA[dataset]}

    # Only include models present in this dataset, in fixed global order
    names      = [m for m in ALL_MODELS if m in rows]
    throughputs = [rows[m][0] for m in names]
    latencies   = [rows[m][1] for m in names]
    sds         = [rows[m][2] for m in names]
    colors      = [COLOR_MAP[m] for m in names]

    x = np.arange(len(names))
    slug = dataset.lower().replace('-', '_')

    throughput_sds = [tp * sd / lat for tp, lat, sd in zip(throughputs, latencies, sds)]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.barh(x, throughputs, xerr=throughput_sds, color=colors, height=0.8,
             error_kw=dict(ecolor="black", capsize=3, linewidth=1.0))
    for i, (tp, sd, col) in enumerate(zip(throughputs, throughput_sds, colors)):
        ax1.text(tp + sd, i, f" {tp/1000:.0f}k", va="center", ha="left", fontsize=8, color=col)
    ax1.set_yticks(x)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Throughput (samples/s)")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax1.margins(x=0.1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(f"throughput_{slug}.pdf", bbox_inches="tight")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.barh(x, latencies, xerr=sds, color=colors, height=0.8,
             error_kw=dict(ecolor="black", capsize=3, linewidth=1.0))
    for i, (lat, sd, col) in enumerate(zip(latencies, sds, colors)):
        ax2.text(lat + sd, i, f" {lat:.1f}", va="center", ha="left", fontsize=8, color=col)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Latency (ms/batch)")
    ax2.margins(x=0.1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(f"latency_{slug}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    for dataset in DATA:
        plot_throughput_latency(dataset)
