import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class ConditionalMotifGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, embedding_dim=8):
        super(ConditionalMotifGCN, self).__init__()
        
        # Standard pairwise message passing
        self.conv_standard = SAGEConv(input_dim, hidden_dim)
        self.conv_standard_out = SAGEConv(hidden_dim, embedding_dim)
        
        # Motif-specific message passing (Triangles with Nodes-of-Interest)
        self.conv_motif = SAGEConv(input_dim, hidden_dim)
        self.conv_motif_out = SAGEConv(hidden_dim, embedding_dim)
        
        # Learned parameter to balance standard edges vs motif edges
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.relu = nn.ReLU()

    def forward(self, x, standard_edge_index, motif_edge_index):
        # Path 1: Standard Topology
        h_std = self.relu(self.conv_standard(x, standard_edge_index))
        h_std = self.conv_standard_out(h_std, standard_edge_index)
        
        # Path 2: Higher-Order Motifs
        h_mtf = self.relu(self.conv_motif(x, motif_edge_index))
        h_mtf = self.conv_motif_out(h_mtf, motif_edge_index)
        
        # Combine embeddings using learned alpha parameter
        alpha_clamped = torch.clamp(self.alpha, 0, 1)
        final_embedding = (alpha_clamped * h_std) + ((1 - alpha_clamped) * h_mtf)
        
        return final_embedding

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(LSTM_VAE, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.reconstruction_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n.squeeze(0)
        mean = self.mean_layer(h_n)
        logvar = self.logvar_layer(h_n)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, seq_len):
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.decoder_lstm(z_repeated)
        reconstruction = self.reconstruction_layer(lstm_out)
        return reconstruction

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z, x.size(1))
        return reconstruction, mean, logvar