import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class AnomalyGAE(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyGAE, self).__init__()
        
        # --- GRAPH ENCODER ---
        # SAGEConv uses edges to pass messages and blend neighbor features
        self.conv1 = SAGEConv(input_dim, 24)
        self.conv2 = SAGEConv(24, 16)
        self.conv3 = SAGEConv(16, 8)
        
        # --- STANDARD DECODER ---
        # Rebuilds the 36 features from the compressed 8-dimensional graph state
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim),
            nn.Sigmoid() # Caps output between 0 and 1
        )
        
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # 1. Message Passing (The GNN Magic)
        # We pass both the node features (x) AND the network structure (edge_index)
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        latent_state = self.relu(self.conv3(x, edge_index))
        
        # 2. Reconstruction
        reconstructed_features = self.decoder(latent_state)
        
        return reconstructed_features