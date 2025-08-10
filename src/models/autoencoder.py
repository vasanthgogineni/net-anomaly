import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, in_dim: int, latent: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, max(64, in_dim//2)),
            nn.ReLU(),
            nn.Linear(max(64, in_dim//2), latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, max(64, in_dim//2)),
            nn.ReLU(),
            nn.Linear(max(64, in_dim//2), in_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

@torch.no_grad()
def reconstruction_error(model: 'AE', x: torch.Tensor) -> torch.Tensor:
    recon = model(x)
    return torch.mean((x - recon) ** 2, dim=1)
