import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tversky import nn as tnn
from utils import config_overridable


class TverskyProj(nn.Module):
    """
    SIRL with Tversky similarity in the triplet loss.

    Encoder is identical to baseline SIRL. The TverskySimilarity module
    (with learnable feature bank Ω and contrast weights) replaces L2
    distance for comparing trajectory embeddings.
    """
    def __init__(self, input_dim=567, latent_dim=6,
                 fbank_size=128, similarity_model='contrast',
                 intersection_reduction='product',
                 difference_reduction='ignorematch',
                 normalize=False):
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
        }
        self.encoder = nn.Sequential(
            tnn.TverskyProjection(
                embedding_dim=input_dim,
                class_count=latent_dim, # prototype count
                fbank_size=fbank_size, # embedding_dim = input_dim
                similarity_model='contrast',
                normalize=False
            )
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.encoder(x)
        
    def save_model(self, path):
        """Save state_dict + constructor hparams so load_model can reconstruct exactly."""
        torch.save({
            'hparams': self.hparams,
            'state_dict': self.state_dict(),
        }, path)
        print(f"model saved to {path}")


def symmetric_triplet_loss(model, a_emb, p_emb, n_emb, margin=1.0):
    """L_trip(A, P, N) + L_trip(P, A, N), using L2 distance (normal triplet loss)."""
    loss_fn = nn.TripletMarginLoss(
        margin=margin, reduction='mean'
    )
    return loss_fn(a_emb, p_emb, n_emb) + loss_fn(p_emb, a_emb, n_emb)


def asymmetric_triplet_loss(model, a_emb, p_emb, n_emb, margin=1.0):
    loss_fn = nn.TripletMarginLoss(
        margin=margin, reduction='mean'
    )
    return loss_fn(a_emb, p_emb, n_emb)


@config_overridable
def train_tversky_proj(
    config,
    anchors, positives, negatives,
    num_epochs=3000,
    batch_size=64,
    lr=0.004,
    lr_decay=0.99999,
    margin=1.0,
    latent_dim=6,
    fbank_size=128,
    similarity_model='contrast',
    intersection_reduction='product',
    difference_reduction='substractmatch',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    use_symmetric_loss=True,
):
    A = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    P = torch.as_tensor(positives, dtype=torch.float32, device=device)
    N = torch.as_tensor(negatives, dtype=torch.float32, device=device)

    assert A.shape == P.shape == N.shape, f"shape mismatch: {A.shape}, {P.shape}, {N.shape}"

    # flatten to 2d
    A = A.flatten(start_dim=1)
    P = P.flatten(start_dim=1)
    N = N.flatten(start_dim=1)

    input_dim = A.shape[1]
    n_triplets = A.shape[0]
    print(f"input dim: {input_dim}")

    model = TverskyProj(
        input_dim=input_dim, latent_dim=latent_dim,
        fbank_size=fbank_size, similarity_model=similarity_model,
        intersection_reduction=intersection_reduction,
        difference_reduction=difference_reduction,
    ).to(device)

    # Adam now optimizes encoder + Tversky feature bank + α/β/θ
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    history = []
    for epoch in range(num_epochs):
        model.train()
        idx = np.random.choice(n_triplets, size=min(batch_size, n_triplets), replace=False)
        idx = torch.as_tensor(idx, device=device)

        a_emb = model(A[idx])
        p_emb = model(P[idx])
        n_emb = model(N[idx])

        if use_symmetric_loss:
            loss = symmetric_triplet_loss(model, a_emb, p_emb, n_emb, margin=margin)
        else:
            loss = asymmetric_triplet_loss(model, a_emb, p_emb, n_emb, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # eval using L2 norm, like SIRL
                ae, pe, ne = model(A), model(P), model(N)
                ap = torch.norm(ae - pe, dim=1)
                an = torch.norm(ae - ne, dim=1)
                pn = torch.norm(pe - ne, dim=1)
                # Symmetric accuracy: both orderings should hold
                acc_a = (an > ap).float().mean().item()
                acc_p = (pn > ap).float().mean().item()
                acc = 0.5 * (acc_a + acc_p)
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} | "
                  f"triplet_acc={acc:.3f} | lr={scheduler.get_last_lr()[0]:.5f}")
            history.append({'epoch': epoch, 'loss': loss.item(), 'acc': acc})

    return model, history


def load_tversky_proj(path, device=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TverskyProj(**ckpt['hparams'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    if device:
        model = model.to(device)
    return model