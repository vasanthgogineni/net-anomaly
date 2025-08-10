import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    """Sequence autoencoder; encodes a sequence of feature vectors and reconstructs it.
    Use the reconstruction error of the last time step as anomaly score.
    """
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x):
        # x: (B, T, F)
        enc_out, (h, c) = self.encoder(x)
        dec_in = enc_out  # simple mirror
        dec_out, _ = self.decoder(dec_in)
        y = self.out(dec_out)  # (B, T, F)
        return y

@torch.no_grad()
def sequence_recon_error(model: 'LSTMAE', x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    err = torch.mean((x[:, -1, :] - y[:, -1, :]) ** 2, dim=1)
    return err
