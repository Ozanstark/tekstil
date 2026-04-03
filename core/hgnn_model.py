import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class HGNNAnomalyDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64):
        super(HGNNAnomalyDetector, self).__init__()
        self.hgc1 = HypergraphConv(in_channels, hidden_channels)
        self.hgc2 = HypergraphConv(hidden_channels, out_channels)
        
        # Projection for anomaly scoring
        self.projection = nn.Linear(out_channels, out_channels)
        
    def forward(self, x, hyperedge_index):
        # x: Node features [N, F]
        # hyperedge_index: [2, E] where row 0 is node_id and row 1 is hyperedge_id
        
        x = self.hgc1(x, hyperedge_index)
        x = F.relu(x)
        features_1 = x
        
        x = self.hgc2(x, hyperedge_index)
        x = F.relu(x)
        features_2 = x
        
        z = self.projection(x)
        return z, features_1, features_2

def incidence_to_edge_index(H):
    """
    Converts incidence matrix H [N, E] to PyTorch Geometric hyperedge_index [2, M]
    where M is the number of non-zero entries in H.
    """
    node_indices, edge_indices = torch.where(H > 0)
    return torch.stack([node_indices, edge_indices], dim=0)
