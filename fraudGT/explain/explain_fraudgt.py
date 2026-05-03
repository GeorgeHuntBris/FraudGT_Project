import torch

import os
import copy

from fraudGT.explain.fraudgt_attention_explainer import FraudGTAttentionExplainer
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explanation

from fraudGT.graphgym.config import cfg, set_cfg, load_cfg
from fraudGT.graphgym.cmd_args import parse_args
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.checkpoint import load_ckpt
from fraudGT.utils import custom_set_out_dir, custom_set_run_dir
from fraudGT.graphgym.loader import create_loader
from fraudGT.graphgym.utils.device import auto_select_device


# Config loading
# Read command line args (specifically, path to model config)
args = parse_args()
# Initialise default config then load yaml on top
set_cfg(cfg)
load_cfg(cfg, args)
custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
cfg.train.auto_resume = True  # prevent custom_set_run_dir from wiping the checkpoint directory
custom_set_run_dir(cfg, cfg.seed)  # load from seed 42 (first repeat)

auto_select_device(strategy='greedy')
if cfg.device == 'auto':
    cfg.device = 'cuda:{}'.format(args.gpu)

# Load dataset & model
loaders, dataset = create_loader(returnDataset=True)
model = create_model(dataset=dataset)

# Load checkpoint (trained model weights)
load_ckpt(model, optimizer=None, scheduler=None)
model.eval() # Put in eval mode (displaces dropout etc)

# Get test loader
test_loader = loaders[2]

# Find a batch with enough seed-node positives
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)
for batch in test_loader:
    batch = batch.to(device)
    seed_positives = batch['node'].y[:batch['node'].batch_size].sum().item()
    print(f"Batch seed positives: {seed_positives}")
    if seed_positives >= 3:
        break

# Identify target node for explanations
pred = model(copy.deepcopy(batch))[0] # Returns (pred, label) - > just take pred. Gives tensor of shape (N,1)
print(f"pred (raw logits) min={pred.min().item():.3f}, max={pred.max().item():.3f}, mean={pred.mean().item():.3f}")

fraud_scores = torch.sigmoid(pred.squeeze()) # convert from raw scores to probabilities
labels = batch['node'].y[:len(fraud_scores)]
n_id = batch['node'].n_id # original node IDs (NeighborLoader saves these)

print(f"batch['node'].batch_size: {batch['node'].batch_size}")
print(f"len(fraud_scores): {len(fraud_scores)}")
print(f"len(batch['node'].y): {len(batch['node'].y)}")
print(f"Total label=1 in full batch y: {batch['node'].y.sum().item()}")
print(f"Total label=1 in sliced labels: {labels.sum().item()}")
print(f"Total predicted positive (score>0.5): {(fraud_scores > 0.5).sum().item()}")

# Group 1: top 10 highest predicted phishing scores
top10_indices = fraud_scores.topk(10).indices

# Group 2: mid-level predictions (score 0.6-0.7; first 10)
mid_mask = (fraud_scores >= 0.6) & (fraud_scores <= 0.7)
mid_indices = mid_mask.nonzero(as_tuple=True)[0][:10] # nonzero returns indices of all non-zero

# Group 3: true positives (label=1 and score > 0.5; first 10)
tp_mask = (labels==1) & (fraud_scores > 0.5)
tp_indices = tp_mask.nonzero(as_tuple=True)[0][:10]

# Group 4: mid-level predictions (score 0.35-0.5; first 10)
low_mask = (fraud_scores >= 0.35) & (fraud_scores <= 0.5)
low_mid_indices = low_mask.nonzero(as_tuple=True)[0][:10] # nonzero returns indices of all non-zero


print("Group 1 - Top 10 highest predictions")
for idx in top10_indices:
    print(f"batch idx {idx.item()}: csv node id: {n_id[idx].item()}, label={batch['node'].y[idx].item()}, score={fraud_scores[idx].item():.3f}")

print("Group 2 - Mid-level predictions (0.6-0.7)")
for idx in mid_indices:
    print(f"batch idx {idx.item()}: csv node id: {n_id[idx].item()}, label={batch['node'].y[idx].item()}, score={fraud_scores[idx].item():.3f}")

print("Group 3 - True positives (label =1, score > 0.5)")
for idx in tp_indices:
    print(f"batch idx {idx.item()}: csv node id: {n_id[idx].item()}, label={batch['node'].y[idx].item()}, score={fraud_scores[idx].item():.3f}")

print("Group 4 - Low-mid-level predictions (0.35-0.5) ")
for idx in low_mid_indices:
    print(f"batch idx {idx.item()}: csv node id: {n_id[idx].item()}, label={batch['node'].y[idx].item()}, score={fraud_scores[idx].item():.3f}")


tp = tp_mask.sum().item()
fp = (fraud_scores > 0.5).sum().item() - tp
fn = labels.sum().item() - tp
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Total predicted positive: {(fraud_scores > 0.5).sum().item()}, Total label=1: {labels.sum().item()}")
print(f"TP={tp}, FP={fp}, FN={fn}")
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

explainer = FraudGTAttentionExplainer(model)
edge_importance, homo_edge_index, homo_edge_type = explainer.explain(batch)
homo_edge_index = homo_edge_index.cpu()
edge_importance = edge_importance.cpu()
homo = copy.deepcopy(batch).to_homogeneous()
x_homo = homo.x.cpu()
num_nodes = homo.num_nodes


# Run Explanations
results_dir = os.path.dirname(os.path.dirname(os.path.abspath(cfg.run_dir)))
exp_dir = os.path.join(results_dir, 'explanations', os.path.basename(cfg.out_dir))
print(f"Saving explanations to: {exp_dir}")
os.makedirs(exp_dir, exist_ok=True)

# Save all predictions to CSV for cross-model comparison
import pandas as pd
preds_df = pd.DataFrame({
    'batch_idx': range(len(fraud_scores)),
    'csv_node_id': n_id[:len(fraud_scores)].cpu().numpy(),
    'label': labels.cpu().numpy(),
    'score': fraud_scores.detach().cpu().numpy()
})
preds_df.to_csv(f'{exp_dir}/predictions.csv', index=False)
print(f"Saved predictions to {exp_dir}/predictions.csv")

for group_name, group_indices in [('top', top10_indices), ('mid', mid_indices), ('tp', tp_indices), ('low-mid', low_mid_indices)]:
    for i, node_idx in enumerate(group_indices):

        # Find 2-hop neighbourhood using directed edges
        subset, _, _, edge_mask_k = k_hop_subgraph(
            node_idx.item(), num_hops=2, edge_index=homo_edge_index,
            relabel_nodes=False, num_nodes=num_nodes
        )

        # Relabel nodes
        relabel = torch.full((num_nodes,), -1, dtype=torch.long)
        relabel[subset] = torch.arange(subset.size(0))

        sub_edge_index = relabel[homo_edge_index[:, edge_mask_k]]
        sub_edge_importance = edge_importance[edge_mask_k]

        # Node labels: batch_idx for all nodes
        node_labels = [str(orig_idx.item()) for orig_idx in subset]

        explanation = Explanation(
            x=x_homo[subset],
            edge_index=sub_edge_index,
            edge_mask=sub_edge_importance,
            y=homo.y[subset] if homo.y is not None else None,
        )

        explanation.n_id = batch['node'].n_id.cpu()  # original csv node IDs (so we can look up any nodes)
        explanation.y_full = batch['node'].y.cpu()   # labels for all batch nodes
        torch.save(explanation, f'{exp_dir}/{group_name}_node_{node_idx}.pt')

        if i < 10 and explanation.edge_index.numel() > 0:
            # Ensure graph is not too dense for clear visualizations
            if explanation.edge_mask is not None and explanation.edge_mask.numel() > 45:
                topk_val, topK_idx = explanation.edge_mask.topk(45)
                mask = torch.zeros_like(explanation.edge_mask)
                mask[topK_idx] = topk_val
                explanation.edge_mask = mask
            explanation.visualize_graph(path=f'{exp_dir}/graph_{group_name}_node_{node_idx}.png', node_labels=node_labels)










