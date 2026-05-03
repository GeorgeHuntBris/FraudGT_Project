import os
import os.path as osp
import shutil
import numpy as np
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DGraphFin
from torch_geometric.utils import index_to_mask

from .temporal_dataset import TemporalDataset


def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def to_adj_nodes_with_times(data):
    num_nodes = data['node'].num_nodes
    timestamps = data['node', 'to', 'node'].timestamps
    if timestamps is None:
        timestamps = torch.zeros(data['node', 'to', 'node'].edge_index.shape[1], 1)
    else:
        timestamps = timestamps.reshape(-1, 1)
    edges = torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = {i: [] for i in range(num_nodes)}
    adj_list_in = {i: [] for i in range(num_nodes)}
    for u, v, t in edges:
        u, v, t = int(u), int(v), int(t)
        adj_list_out[u].append((v, t))
        adj_list_in[v].append((u, t))
    return adj_list_in, adj_list_out


def ports(edge_index, adj_list):
    ports_tensor = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1:
            continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:, [0]], return_index=True, axis=0)
        nbs_unique = a[np.sort(idx)][:, 0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u, v)] = i
    for i, e in enumerate(edge_index.T):
        ports_tensor[i] = ports_dict[tuple(e.numpy())]
    return ports_tensor


class DGraphDataset(TemporalDataset):
    """
    Download DGraphFin.zip from https://dgraph.xinye.com/dataset
    DGraph has the built-in masks already that come with the DGraphFin dataset. So already in train/val/test.
    Literally has a built-in Dgraph class which is amazing.
    """

    def __init__(
        self,
        root: str,
        reverse_mp: bool = False,
        add_ports: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = 'DGraph'
        self.reverse_mp = reverse_mp
        self.add_ports = add_ports
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0], weights_only=False)

        if not reverse_mp:
            for split in ['train', 'val', 'test']:
                if ('node', 'rev_to', 'node') in self.data_dict[split].edge_types:
                    del self.data_dict[split]['node', 'rev_to', 'node']

        if add_ports:
            self.ports_dict = torch.load(self.processed_paths[1], weights_only=False)
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.add_ports_func(
                    self.data_dict[split], self.ports_dict[split]
                )

    def add_ports_func(self, data, ports_data):
        in_ports, out_ports = ports_data
        if not self.reverse_mp:
            data['node', 'to', 'node'].edge_attr = torch.cat(
                [data['node', 'to', 'node'].edge_attr, in_ports, out_ports], dim=1
            )
        else:
            data['node', 'to', 'node'].edge_attr = torch.cat(
                [data['node', 'to', 'node'].edge_attr, in_ports], dim=1
            )
            data['node', 'rev_to', 'node'].edge_attr = torch.cat(
                [data['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1
            )
        return data

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['DGraphFin.zip']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'ports.pt']

    def download(self):
        raise FileNotFoundError(
            f"DGraphFin.zip not found in {self.raw_dir}.\n"
            "Please download DGraphFin.zip from https://dgraph.xinye.com/dataset\n"
            f"and place it at: {osp.join(self.raw_dir, 'DGraphFin.zip')}"
        )

    def process(self):
        print("Loading DGraph-Fin dataset via PyG...")

        # Use a separate subdirectory for PyG's DGraphFin to avoid
        # conflicts with our own processed files in processed dir
        pyg_root = osp.join(self.root, 'dgraph_pyg') # Build data/DGraph/dgraph_pyg path
        os.makedirs(osp.join(pyg_root, 'raw'), exist_ok=True) # Create folder data/DGraph/dgraph_pyg/raw/


        zip_src = osp.join(self.raw_dir, 'DGraphFin.zip') # Builds the path to where manually placed hte zip file
        zip_dst = osp.join(pyg_root, 'raw', 'DGraphFin.zip') # Builds dest path to where PyG expects to find it.
        if not osp.exists(zip_dst):
            shutil.copy(zip_src, zip_dst) # Only copies if one doesn't already exit there.

        # pyg_dataset is effectively a list containing one graph -> get that one item.
        pyg_dataset = DGraphFin(root=pyg_root) # Create a PyG DGraphFin dataset object (extracts zip and reads raw files and saves)
        raw = pyg_dataset[0] # Get the dataset (1 graph)

        print(f"Number of nodes: {raw.num_nodes}")
        print(f"Number of edges: {raw.edge_index.shape[1]}")
        print(f"Node feature shape: {raw.x.shape}")

        # Node features (17-dim) normalised
        x = z_norm(raw.x.float())
        y = raw.y.squeeze().long() # [num_nodes, 1] -> get rid of extra dim [,1]

        # Use provided train/val/test masks from DGraphFin
        train_mask = raw.train_mask
        val_mask = raw.val_mask
        test_mask = raw.test_mask

        # Background nodes (not in any split) have labels > 1 — mask them to -1
        # so the framework sees only binary labels (0=normal, 1=fraud)
        foreground_mask = train_mask | val_mask | test_mask
        y[~foreground_mask] = -1 # Incase Background nodes could be 2,3, 4 etc. so any node not in the masks put background label as -1.

        # Edge features: combine edge_type [E] and edge_time [E] into [E, 2]. This is necessary as coming requires tenors to be 2d to stack them as collumns
        # Expected format is that edge_att contains edge type and the features (its just another feature of hte edge_attr)
        edge_type = raw.edge_type.float().unsqueeze(1) # Get edge types [4.3M, 1]
        edge_time = raw.edge_time.float().unsqueeze(1) # Timestamp for each edge reshaped into [4.3M, 1]
        edge_attr = z_norm(torch.cat([edge_type, edge_time], dim=1))

        edge_index = raw.edge_index
        timestamps = raw.edge_time.float()

        print(f"Train nodes: {train_mask.sum().item()}")
        print(f"Val nodes: {val_mask.sum().item()}")
        print(f"Test nodes: {test_mask.sum().item()}")

        # Labels in train of licit & ilicit
        train_labeled = y[train_mask]
        illicit = (train_labeled == 1).sum().item()
        licit = (train_labeled == 0).sum().item()
        print(f"Train illicit: {illicit}, licit: {licit}, ratio=1:{licit // illicit}")

        # All splits use the full graph — DGraph is transductive and background nodes
        # (66.8% of all nodes) are essential for connectivity.
        num_nodes = raw.num_nodes
        self.ports_dict = {}
        self.data_dict = {}

        for split in ['train', 'val', 'test']:
            split_mask = eval(f'{split}_mask')

            data = HeteroData()
            data['node'].x = x
            data['node'].y = y
            data['node'].num_nodes = num_nodes
            data['node'].train_mask = train_mask
            data['node'].val_mask = val_mask
            data['node'].test_mask = test_mask
            data['node'].split_mask = split_mask
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            data['node', 'to', 'node'].edge_index = edge_index
            data['node', 'to', 'node'].edge_attr = edge_attr
            data['node', 'to', 'node'].timestamps = timestamps

            # Adds reverse edges to the graph so the model can do rmp
            data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()   # Flip the source and dst nodes
            data['node', 'rev_to', 'node'].edge_attr = edge_attr # Reverse edge features get the same edge features as the forward

            adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            out_ports = ports(data['node', 'to', 'node'].edge_index.flipud(), adj_list_out)

            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data

        if self.pre_transform is not None:
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.pre_transform(self.data_dict[split])

        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])
        print("Processing complete!")

    def __repr__(self) -> str:
        return f'DGraphDataset(name={self.name})'
