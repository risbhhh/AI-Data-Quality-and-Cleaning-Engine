import torch
import torch.nn as nn

class TabularAutoencoder(nn.Module):
    def __init__(self, n_features:int, hidden_dim:int=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(4, hidden_dim//2)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(4, hidden_dim//2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
