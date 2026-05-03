import json, os
import numpy as np

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

BATCH_SIZE = 2048  # same across all models/datasets


# Takes a models result folder and returns the test metrics at the best validation epoch.
def get_metrics(model_path, max_seeds=8, best_metric="f1"):
    seed_dirs = sorted([d for d in os.listdir(model_path) if d.isdigit()])[:max_seeds] # Keep only seed folders form model subdirectories (purer numbers)
    # Create an empty dict where each metric name maps to empty list (values get appended as loop through seeds)
    metrics = {k: [] for k in ["f1", "precision", "recall", "accuracy", "macro_f1", "auc", "ap",
                                "test_f1", "test_precision", "test_recall", "test_accuracy", "test_auc", "test_ap"]}

    for seed in seed_dirs:
        # For each seed folder, build the paths to the val and test file stats
        val_path = os.path.join(model_path, seed, "val", "stats.json")
        test_path = os.path.join(model_path, seed, "test", "stats.json")
        if not os.path.exists(val_path) or not os.path.exists(test_path):
            continue
        # Read file line by line
        val_stats = [json.loads(l) for l in open(val_path)]
        # Find the largest element in the list. compare by dicts by key, (f1 default)
        best = max(val_stats, key=lambda x: x.get(best_metric, 0))
        best_epoch = best["epoch"]
        # Pull metrics from best single epoch
        metrics["f1"].append(best.get("f1", 0))
        metrics["precision"].append(best.get("precision", 0))
        metrics["recall"].append(best.get("recall", 0))
        metrics["accuracy"].append(best.get("accuracy", 0))
        metrics["macro_f1"].append(best.get("macro-f1", 0))
        metrics["auc"].append(best.get("auc", 0))
        metrics["ap"].append(best.get("ap", 0))
        # Read test stats file (give a list of dicts, one per epoch)
        test_stats = [json.loads(l) for l in open(test_path)]
        # Search through and find epoch corresponding to best val epoch
        t = next((s for s in test_stats if s["epoch"] == best_epoch), None)
        # If not found -> fall back on best test epoch instead
        if t is None:
            t = max(test_stats, key=lambda x: x.get(best_metric, 0))
        metrics["test_f1"].append(t.get("f1", 0))
        metrics["test_precision"].append(t.get("precision", 0))
        metrics["test_recall"].append(t.get("recall", 0))
        metrics["test_accuracy"].append(t.get("accuracy", 0))
        metrics["test_auc"].append(t.get("auc", 0))
        metrics["test_ap"].append(t.get("ap", 0))
    return metrics


def get_throughput_latency(model_path, max_seeds=5):
    """
    Compute inference throughput (trans/s) and per-batch latency (ms).
    Methodology from FraudGT paper Fig 3:
      - Latency l = average per-batch inference time (time_iter from val stats)
      - Throughput = T / l, where T = batch_size
     average time_iter across all val epochs and seeds.
    """
    seed_dirs = sorted([d for d in os.listdir(model_path) if d.isdigit()])[:max_seeds] # Sorts seeds and take max seeds
    all_time_iters = [] # Accumulate every time_iter value across all seeds and epochs
    for seed in seed_dirs:
        # Loop through seeds and construct path to val stats file
        # Val chosen over test as 100 epochs per seed compared to jsut 1 for test -> more accurate
        val_path = os.path.join(model_path, seed, "val", "stats.json")
        if not os.path.exists(val_path):
            continue
        val_stats = [json.loads(l) for l in open(val_path)] # Read file line by line - each one contains a stats.json for one epoch worht of stats (so 100 per seed)
        # Skip first few epochs (warm-up) if enough data
        if len(val_stats) > 10:
            val_stats = val_stats[5:]
        for s in val_stats:
            if s.get("time_iter", 0) > 0:
                all_time_iters.append(s["time_iter"])
    # If no valid time_iter were collected -> return None immediately
    if not all_time_iters:
        return None, None, None, None
    latency_s = np.mean(all_time_iters)
    latency_std = np.std(all_time_iters)
    throughput = BATCH_SIZE / latency_s
    latency_ms = latency_s * 1000
    latency_ms_std = latency_std * 1000
    return throughput, latency_ms, latency_ms_std, len(all_time_iters)




# format: takes a list of values (one per seed)
def fmt(vals):
    if not vals:
        return "   -       "
    return f"{np.mean(vals):.4f} ±{np.std(vals):.4f}"

# Print results table. Takes a prefix (dataset (e.g. Eth)) to filter which models to include,
# scans the results directory for matching model folders, calls get_metrics on each one, and prints a formatted
# table of val and test metrics side by side.
def analyse(prefix, best_metric="f1"):
    models = sorted([d for d in os.listdir(results_dir) if d.startswith(prefix)])

    # Print the table header (with necessary padding)
    print(f"\n{'Model':<30} {'F1 (Val)':<16} {'Prec (Val)':<16} {'Rec (Val)':<16} {'AUC (Val)':<16} {'AP (Val)':<16}  |  {'F1 (Test)':<16} {'Prec (Test)':<16} {'Rec (Test)':<16} {'AUC (Test)':<16} {'AP (Test)':<16}")
    print("-" * 215)

    for model in models:
        # Loop through models and build full path to its results folder
        model_path = os.path.join(results_dir, model)
        # Get all the val and test metric lists for that model
        m = get_metrics(model_path, best_metric=best_metric)
        # Clean folder name for display
        name = model.replace("-gpu0", "").replace(prefix.rstrip("-"), "").lstrip("-")
        # If at least one seed completed, print the full row with all metrics formatted otherwise no results.
        if m["f1"]:
            print(f"{name:<30} {fmt(m['f1']):<16} {fmt(m['precision']):<16} {fmt(m['recall']):<16} "
                  f"{fmt(m['auc']):<16} {fmt(m['ap']):<16}  |  {fmt(m['test_f1']):<16} {fmt(m['test_precision']):<16} "
                  f"{fmt(m['test_recall']):<16} {fmt(m['test_auc']):<16} {fmt(m['test_ap']):<16}")
        else:
            print(f"{name:<30} no results")


def analyse_throughput(prefix):
    """Print throughput and latency table for all models with a given prefix."""
    models = sorted([d for d in os.listdir(results_dir) if d.startswith(prefix)]) # get all model directories s(e.g. elliptic-fraudgt, elliptic-gie)
    results = [] # One tuple per model
    for model in models:
        model_path = os.path.join(results_dir, model)
        tp, lat_ms, lat_std, n = get_throughput_latency(model_path)
        name = model.replace("-gpu0", "").replace(prefix.rstrip("-"), "").lstrip("-") # Clean directory names for readability
        results.append((name, tp, lat_ms, lat_std))

    # Sort by throughput descending
    results.sort(key=lambda x: x[1] if x[1] else 0, reverse=True) # Sort the results list by index 1 (throughput - highest first)

    print(f"\n{'Model':<30} {'Throughput (trans/s)':<25} {'Latency (ms/batch)':<25}")
    print("-" * 80)
    for name, tp, lat_ms, lat_std in results:
        if tp is not None:
            print(f"{name:<30} {tp:<25.1f} {lat_ms:.1f} ±{lat_std:.1f}")
        else:
            print(f"{name:<30} no results")



# ACCURACY RESULTS
print("=" * 185)
print("ELLIPTIC RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("Elliptic-", best_metric="f1")

print()
print("=" * 185)
print("ETH RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("ETH-", best_metric="f1")

print()
print("=" * 185)
print("DGRAPH RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("DGraph-", best_metric="f1")

print()
print("=" * 185)
print("BITCOIN-M RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("BitcoinM-", best_metric="f1")

print()
print("=" * 185)
print("ETHEREUM-P RESULTS (Val metrics @ best val F1 epoch, up to 5 seeds)")
print("=" * 185)
analyse("EthereumP-", best_metric="f1")


# THROUGHPUT AND LATENCY  (FraudGT paper Fig 3 methodology)
# Throughput = batch_size / mean_per_batch_inference_time  (trans/s)
# Latency    = mean_per_batch_inference_time               (ms/batch)

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — Elliptic")
print("=" * 80)
analyse_throughput("Elliptic-")

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — ETH")
print("=" * 80)
analyse_throughput("ETH-")

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — DGraph")
print("=" * 80)
analyse_throughput("DGraph-")

print()
print("=" * 80)
print("THROUGHPUT AND LATENCY — Bitcoin-M")
print("=" * 80)
analyse_throughput("BitcoinM-")



