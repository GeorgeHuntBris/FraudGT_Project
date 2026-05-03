

import os
import os.path as osp
import pandas as pd
import numpy as np
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import index_to_mask

from .temporal_dataset import TemporalDataset


"""
Elliptic comes as three separate csv files rather than a pre-built .pt file. 

elliptic_txs_features.csv -> one row per transaction node. col[0] = transaction ID, col[1] = timestep, then features..
elliptic_txs_edgelist.csv -> one row per edge. Cols are source transaction, dest transaction 
elliptic_txs_classes.csv -> one row per labeled transaction. 2 cols; transaction ID and class

Map transaction IDs from Ellpitic format to my format
Map class labels form ellpitic format to my format
Recreate mapping between transformed ids and labels

"""

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

#  Build adjacency lists with timestamps for port computation
def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = (
        torch.zeros((data.edge_index.shape[1], 1))
        if data['node', 'to', 'node'].timestamps is None
        else data['node', 'to', 'node'].timestamps.reshape((-1, 1))
    )
    edges = torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u, v, t in edges:
        u, v, t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out


def ports(edge_index, adj_list):
    """Compute port numberings for edges based on temporal ordering."""
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


class EllipticDataset(TemporalDataset):


    # Train on time steps 1-34, validate on 35-38, test on 39-42
    # Excludes timesteps 43-49 (post dark market shutdown at timestep 43)
    TRAIN_TIME_STEPS = list(range(1, 35))   # 34 time steps
    VAL_TIME_STEPS = list(range(35, 39))    # 4 time steps
    TEST_TIME_STEPS = list(range(39, 43))   # 4 time steps

    def __init__(
        self,
        root: str,
        name: str = 'elliptic',
        reverse_mp: bool = False,
        add_ports: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        self.name = name
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
        """Add port numbering features to edge attributes."""
        in_ports, out_ports = ports_data

        if not self.reverse_mp:
            out_ports_list = [out_ports]
            data['node', 'to', 'node'].edge_attr = torch.cat(
                [data['node', 'to', 'node'].edge_attr, in_ports] + out_ports_list, dim=1
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
        return [
            'elliptic_txs_features.csv',
            'elliptic_txs_edgelist.csv',
            'elliptic_txs_classes.csv'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'ports.pt']

    def download(self):
        for filename in self.raw_file_names:
            filepath = osp.join(self.raw_dir, filename)
            if not osp.exists(filepath):
                raise FileNotFoundError(
                    f"File '{filename}' not found in {self.raw_dir}. "
                    "Please download from Kaggle."
                )

    def process(self):
        """Process raw Elliptic data into PyG HeteroData format."""

        # Load raw data
        print("Loading Elliptic dataset...")
        # Load node features into pandas frame: txId, time_step, then 166 features (pands dataframes is sthe standard for working with csv)
        df_features = pd.read_csv(
            osp.join(self.raw_dir, 'elliptic_txs_features.csv'),
            header=None
        )

        # Load edge list CSV into pandas data frame - one row per directed edge between two transactions
        df_edges = pd.read_csv(
            osp.join(self.raw_dir, 'elliptic_txs_edgelist.csv')
        )

        # Read classes
        df_classes = pd.read_csv(
            osp.join(self.raw_dir, 'elliptic_txs_classes.csv')
        )

        # Get all transaction ids and map to numbers and idx them
        # Recreate simple ideas based on the transaction ids (e.g. 1,2,3..)
        all_nodes = df_features[0].values
        node_id_map = {txid: idx for idx, txid in enumerate(all_nodes)}  #(dict that maps ellpitic ids to my new ids)
        num_nodes = len(all_nodes)

        # Extract time steps and features
        time_steps = df_features[1].values  # Column 1 is time step
        node_features = df_features.iloc[:, 2:].values  # Columns 2-167 are features

        print(f"Number of nodes (transactions): {num_nodes}")
        print(f"Number of features per node: {node_features.shape[1]}")
        print(f"Time steps: {int(time_steps.min())} to {int(time_steps.max())}")

        # Process labels. Remap elliptic label convention int the framework for code base (0 = licit, 1 = illicit, -1 = unknown)
        # Map: 1 -> 1 (illicit), 2 -> 0 (licit), "unknown" -> -1 (mask out)
        df_classes['mapped_class'] = df_classes['class'].apply(
            lambda x: 1 if x == '1' else (0 if x == '2' else -1)
        )

        # Recreate mapping between transformed ids and transformed class labels and store in labels.
        # Create label array aligned with node indices
        labels = -1 * np.ones(num_nodes, dtype=np.int64) # Create a np array of length of nodes filled entirely with -1. This is the starting point - every node is assumed unlabeleld until proven otherwise.
        for _, row in df_classes.iterrows():
            txid = row['txId']
            if txid in node_id_map:
                labels[node_id_map[txid]] = row['mapped_class']

        # Compute stats about label dist
        labeled_mask = labels != -1
        n_labeled = labeled_mask.sum()
        n_illicit = (labels == 1).sum()
        n_licit = (labels == 0).sum()

        print(f"Labeled nodes: {n_labeled} / {num_nodes} ({100*n_labeled/num_nodes:.1f}%)")
        print(f"Illicit: {n_illicit} ({100*n_illicit/n_labeled:.2f}% of labeled)")
        print(f"Licit: {n_licit} ({100*n_licit/n_labeled:.2f}% of labeled)")


        # Process edges - map to consecutive node indices
        # Map edges using the mapping dict
        edge_src = df_edges['txId1'].map(node_id_map).values
        edge_dst = df_edges['txId2'].map(node_id_map).values

        print(f"Number of edges: {len(edge_src)}")

        # Convert to tensors as working with pytorch tesnors
        x = torch.tensor(node_features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
        timestamps = torch.tensor(time_steps, dtype=torch.float32)

        # Normalize features
        x = z_norm(x)

        # This synthetic timestamp is created as Elliptic raw data has no edge features but  codebase expects stuff -> so manufacture.
        # Create edge timestamps based on source node time step (nodes are transacions)
        edge_timestamps = timestamps[edge_index[0]]
        # Create edge attributes (just timestamp for now, future work explore adding more...)
        edge_attr = edge_timestamps.unsqueeze(1) # add extra dimension to the tensor for the timestamp info

        # Split by time steps
        train_mask = torch.tensor(
            np.isin(time_steps, self.TRAIN_TIME_STEPS), dtype=torch.bool # np.isin(a,b) checks for each element in a whether it appears in b. it retursn a boolean array the same length as a.
        )
        val_mask = torch.tensor(
            np.isin(time_steps, self.VAL_TIME_STEPS), dtype=torch.bool
        )
        test_mask = torch.tensor(
            np.isin(time_steps, self.TEST_TIME_STEPS), dtype=torch.bool
        )

        # Also mask out unknown labels (only determine seed nodes (used for loss and metrics))
        labeled_mask_tensor = torch.tensor(labeled_mask, dtype=torch.bool) # As other masks is already a pytorch tensor
        train_mask = train_mask & labeled_mask_tensor # (only labeled nodes in the train)
        val_mask = val_mask & labeled_mask_tensor
        test_mask = test_mask & labeled_mask_tensor

        print(f"\nSplit statistics (labeled nodes only):")
        print(f"Train: {train_mask.sum().item()} nodes (time steps 1-34)")
        print(f"Val: {val_mask.sum().item()} nodes (time steps 35-42)")
        print(f"Test: {test_mask.sum().item()} nodes (time steps 43-49)")

        # All splits use the full graph — Elliptic is transductive and ~79% of node
        # are unlabeled but essential for connectivity. Each time step is an isolated
        # component so there is no temporal leakage from keeping all edges.
        # Build data for each split
        self.ports_dict = {}
        self.data_dict = {}

        # 3 identical graphs but with different masks
        # Training loop expects one self-contained data object per phrase
        # It uses the split_mask to know which nodes to compute loss on
        for split in ['train', 'val', 'test']:
            split_mask = eval(f'{split}_mask')

            data = HeteroData()
            data['node'].x = x
            data['node'].y = y
            data['node'].num_nodes = num_nodes

            # Store masks for this split (in node store)
            data['node'].train_mask = train_mask
            data['node'].val_mask = val_mask
            data['node'].test_mask = test_mask
            data['node'].split_mask = split_mask  # Current split's mask

            # After each epoch during training it has to compute the val metrics (each graph has all 3 masks - just the current split mask is different).
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

            # Edge data
            data['node', 'to', 'node'].edge_index = edge_index
            data['node', 'to', 'node'].edge_attr = edge_attr
            data['node', 'to', 'node'].timestamps = edge_timestamps

            # RMP
            data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = edge_attr

            # Compute ports
            adj_list_in, adj_list_out = to_adj_nodes_with_times(data)
            in_ports = ports(data['node', 'to', 'node'].edge_index, adj_list_in)
            out_ports = ports(data['node', 'to', 'node'].edge_index.flipud(), adj_list_out)

            self.ports_dict[split] = [in_ports, out_ports]
            self.data_dict[split] = data

        if self.pre_transform is not None:
            for split in ['train', 'val', 'test']:
                self.data_dict[split] = self.pre_transform(self.data_dict[split])

        # Save processed data
        torch.save(self.data_dict, self.processed_paths[0])
        torch.save(self.ports_dict, self.processed_paths[1])

        print("\nProcessing complete!")

    def __repr__(self) -> str:
        return f'EllipticDataset(name={self.name})'
