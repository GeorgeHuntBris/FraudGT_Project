import copy
import torch
from fraudGT.layer.gt_layer import GTLayer

class FraudGTAttentionExplainer:
    """"
    Attention-based explainer for FraudGT
    For each edge, computes importance = max over layers and heads of:
       alpha * mean_D(gate)
    Where:
       alpha (H, num_edges) = post-softmax attention weight
       gate (H, num_edges, D) = sigmoid message passing gate
    Both are saved by GTLayer during the forward pass.
    """
    def __init__(self, model):
        self.model = model
        # Collect all GTLayer modules in the model
        self.gt_layers = [m for m in model.modules() if isinstance(m, GTLayer)]
        assert len(self.gt_layers) > 0, "No GTLayer found in the model"
        print(f"Found {len(self.gt_layers)} GTLayer(s)")

    def explain(self, batch):
        """"
        Run a forward pass and collect edge importance scores
        Args:
            batch: HeteroData batch (copied to avoid issues)
        Returns:
            edge_importance: (num_edges,) one scaler per edge
            homo_edge_index: (2, num_edges)
            homo_edge_type: (num_edges,) 0 = 'to', 1 = 'rev_to'
        """
        self.model.eval() # Disables dropout etc

        # Convert batch to homo as GTLayer internally does this during forward pass
        # Ensures the edge_index matches internal workings of GTLayers
        homo_ref = copy.deepcopy(batch).to_homogeneous() # Don't want to mutate the original graph
        homo_edge_index = homo_ref.edge_index  # (2, num_edges)
        homo_edge_type = homo_ref.edge_type  # (num_edges,) — 0=to, 1=rev_to

        with torch.no_grad(): # Inference -> don't track gradients
            self.model(copy.deepcopy(batch)) # Run forward pass through the full model.

        # Collect importance from each layer
        layer_importances = []
        for layer in self.gt_layers:
            alpha = layer._alpha  # (H, num_edges) — attention weights after softmax
            gate = layer.gate  # (H, num_edges, D) — sigmoid gate values

            gate_scalar = gate.mean(dim=-1)  # (H, num_edges) — collapse D by mean
            importance = alpha * gate_scalar  # (H, num_edges) — combine attention and gate
            layer_importances.append(importance)

        # Stack across layers into single tensor: (L, H, num_edges)
        all_importance = torch.stack(layer_importances, dim=0)

        # Max over L then H → (num_edges,)
        # 1st: for each head and edge -> keep the highest importance score across all heads
        # 2nd: For each edge, keep the highest importance score across all heads
        edge_importance = all_importance.max(dim=0)[0].max(dim=0)[0]

        return edge_importance, homo_edge_index, homo_edge_type






