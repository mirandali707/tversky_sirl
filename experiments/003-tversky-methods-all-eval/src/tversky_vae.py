import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tversky import nn as tnn
from utils import config_overridable


class Decoder(nn.Module):
    """Minimal MLP decoder: latent -> reconstructed flat trajectory."""
    def __init__(self, z_dim, out_dim, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h), nn.GELU(),
            nn.Linear(h, h),     nn.GELU(),
            nn.Linear(h, out_dim),   # linear head, no activation
        )

    def forward(self, z):
        return self.net(z)


class TverskyVAE(nn.Module):
    """
    unsupervised autoencoder with Tversky projection encoder
    """
    def __init__(self, input_dim=19, latent_dim=6,
                 fbank_size=128, similarity_model='contrast',
                 intersection_reduction='product',
                 difference_reduction='ignorematch',
                 normalize=False, decoder_hidden=128):
        super().__init__()
        # store constructor args so save_model/load can reconstruct exactly
        self.hparams = {
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'fbank_size': fbank_size,
            'similarity_model': similarity_model,
            'intersection_reduction': intersection_reduction,
            'difference_reduction': difference_reduction,
            'normalize': normalize,
            'decoder_hidden': decoder_hidden,
        }
        self.encoder = nn.Sequential(
            tnn.TverskyProjection(
                embedding_dim=input_dim,
                class_count=latent_dim,   # prototype count == latent dim
                fbank_size=fbank_size,
                similarity_model=similarity_model,
                normalize=normalize,
            )
        )
        self.decoder = Decoder(
            z_dim=latent_dim, out_dim=input_dim, h=decoder_hidden
        )

    def encode(self, x):
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        return self.decode(z), z

    def save_model(self, path):
        """Save state_dict + constructor hparams so load_model can reconstruct exactly."""
        torch.save({
            'hparams': self.hparams,
            'state_dict': self.state_dict(),
        }, path)
        print(f"model saved to {path}")


@config_overridable
def train_tversky_vae(
    config,
    data,                       # (N, input_dim) or (N, T, D); unlabelled
    num_epochs=3000,
    batch_size=64,
    lr=0.004,
    lr_decay=0.99999,
    latent_dim=6,
    fbank_size=128,
    similarity_model='contrast',
    intersection_reduction='product',
    difference_reduction='substractmatch',
    decoder_hidden=128,
    val_frac=0.1,
    seed=0,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
):
    g = np.random.default_rng(seed)

    X = torch.as_tensor(data, dtype=torch.float32, device=device)
    X = X.flatten(start_dim=1)
    n, input_dim = X.shape
    print(f"n={n} | input dim: {input_dim}")

    # standardize (fit on train split only)
    perm = g.permutation(n)
    n_val = int(round(val_frac * n))
    val_idx = torch.as_tensor(perm[:n_val], device=device)
    train_idx = torch.as_tensor(perm[n_val:], device=device)

    mu = X[train_idx].mean(dim=0, keepdim=True)
    sd = X[train_idx].std(dim=0, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd

    X_train, X_val = X[train_idx], X[val_idx]
    n_train = X_train.shape[0]

    model = TverskyVAE(
        input_dim=input_dim, latent_dim=latent_dim,
        fbank_size=fbank_size, similarity_model=similarity_model,
        intersection_reduction=intersection_reduction,
        difference_reduction=difference_reduction,
        decoder_hidden=decoder_hidden,
    ).to(device)
    # keep normalization stats on the model so inference can reuse them
    model.register_buffer('x_mean', mu)
    model.register_buffer('x_std', sd)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    history = []
    for epoch in range(num_epochs):
        model.train()
        idx = g.choice(n_train, size=min(batch_size, n_train), replace=False)
        idx = torch.as_tensor(idx, device=device)

        xb = X_train[idx]
        xhat, _ = model(xb)
        loss = F.mse_loss(xhat, xb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                tr_hat, z_tr = model(X_train)
                tr_mse = F.mse_loss(tr_hat, X_train).item()
                if n_val > 0:
                    va_hat, _ = model(X_val)
                    va_mse = F.mse_loss(va_hat, X_val).item()
                else:
                    va_mse = float('nan')
                # how many latent dims are actually carrying signal
                z_std = z_tr.std(dim=0)
                active = (z_std > 0.01).sum().item()
            print(f"Epoch {epoch:4d} | train_mse={tr_mse:.4f} | "
                  f"val_mse={va_mse:.4f} | active_dims={active}/{latent_dim} | "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")
            history.append({
                'epoch': epoch,
                'train_mse': tr_mse,
                'val_mse': va_mse,
                'active_dims': active,
                'z_std': z_std.cpu().numpy(),
            })

    return model, history


def load_tversky_vae(path, device=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TverskyVAE(**ckpt['hparams'])
    # buffers are in the state_dict; make sure they exist before loading
    d = ckpt['hparams']['input_dim']
    model.register_buffer('x_mean', torch.zeros(1, d))
    model.register_buffer('x_std', torch.ones(1, d))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    if device:
        model = model.to(device)
    return model