import torch
import torch.nn as nn

class AnomalyAE(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAE, self).__init__()
        
        # Encoder: Compresses 36 features -> 24 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 8) 
        )
        
        # Decoder: Rebuilds 8 -> 16 -> 24 -> 36 features
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim),
            nn.Sigmoid() # Caps the output between 0 and 1 to match MinMaxScaler
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x