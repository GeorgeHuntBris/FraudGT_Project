"""
Compute ETH node degree distribution statistics, split by label.
"""

import torch

data_dict = torch.load('data/ETH/processed/data.pt', weights_only=False)
data = data_dict['test']
edge_index = data['node', 'to', 'node'].edge_index
y = data['node'].y
num_nodes = data['node'].num_nodes

out_degree = torch.bincount(edge_index[0], minlength=num_nodes)
in_degree  = torch.bincount(edge_index[1], minlength=num_nodes)

print(f'Total nodes: {num_nodes:,}')
print(f'Total edges: {edge_index.shape[1]:,}')

for label, name in [(0, 'Normal'), (1, 'Phishing')]:
    mask = (y == label)
    n = mask.sum().item()
    print(f'\n--- {name} nodes (n={n:,}) ---')
    print(f'  Mean out-degree: {out_degree[mask].float().mean().item():.2f}')
    print(f'  Mean in-degree:  {in_degree[mask].float().mean().item():.2f}')
    print(f'  Median out-degree: {out_degree[mask].float().median().item():.1f}')
    print(f'  Median in-degree:  {in_degree[mask].float().median().item():.1f}')
    print(f'  Max out-degree: {out_degree[mask].max().item():,}')
    print(f'  Max in-degree:  {in_degree[mask].max().item():,}')
