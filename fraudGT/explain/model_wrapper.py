import torch
import torch.nn as nn

"""
Accepts plain homogeneous tensors (x, edge_index, edge_attr) from the explainer, skips the encoder                                                                   
(already run once upfront in explain_pna.py), runs just the conv layers + head, and returns just                                                                     
the prediction tensor. Explainer takes the homo path so set_masks can hook into the PNAConv layers                                                                   
and provide real structural edge mask gradients.                                                                                                                     
"""
class HomoGNNExplainerWrapper(nn.Module):
    def __init__(self, model, encoded_batch, node_type_tensor, edge_type_tensor):
        super().__init__()
        self.model = model # We need access to individual components (conv layers,  head etc)
        self.batch = encoded_batch
        self.node_type_tensor = node_type_tensor
        self.edge_type_tensor = edge_type_tensor
        # Save copy of the encoded node features for each node type. .detach() cuts them free of comp. graph so treated as constants
        self.original_x = {nt: encoded_batch[nt].x.detach() for nt in encoded_batch.node_types}

    def forward(self, x, edge_index, edge_attr=None):
        # Restore encoded node features on the batch
        for node_type, feat in self.original_x.items():
            self.batch[node_type].x = feat.clone()

        x = self.model.input_drop(x) # Randomly zero some features during training to prevent overfitting

        src, dst = edge_index
        # Run convolution layers
        for i in range(len(self.model.convs)):
            x = self.model.convs[i](x, edge_index, edge_attr)
            if self.model.layer_norm or self.model.batch_norm: # Normalize node features after each conv layer
                x = self.model.norms[i](x)
            x = self.model.drop(self.model.activation(x)) # Apply ReLU and dropout to node features
            if self.model.edge_updates: # EU updates
                edge_attr = edge_attr + self.model.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        # After conv layers, prediction head reads form batch['node'].x not from a falt tensor -> reconstruct embeddings for this
        for idx, node_type in enumerate(self.batch.node_types): # Loop over all node types in batch
            node_mask = self.node_type_tensor == idx # Construct boolean mask - True for every position in x that belongs to that node type
            self.batch[node_type].x = x[node_mask]

        # Prediction head
        pred, _ = self.model.post_mp(self.batch)
        return pred