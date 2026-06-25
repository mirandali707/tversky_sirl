import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tversky import nn as tnn


class TverskySIRL(nn.Module):
    """
    SIRL with Tversky similarity in the triplet loss.

    Encoder is identical to baseline SIRL. The TverskySimilarity module
    (with learnable feature bank Ω and contrast weights) replaces L2
    distance for comparing trajectory embeddings.
    """
    def __init__(self, input_dim=567, hidden_dim=1024, latent_dim=6,
                 fbank_size=4, similarity_model='contrast',
                 intersection_reduction='product',
                 difference_reduction='ignorematch',
                 normalize=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.tversky_sim = tnn.TverskySimilarity(
            embedding_dim=latent_dim,
            fbank_size=fbank_size,
            similarity_model=similarity_model,
            normalize=normalize,
            intersection_reduction=intersection_reduction,
            difference_reduction=difference_reduction
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.encoder(x)

    def distance(self, a, b):
        """Tversky-based distance: lower = more similar (negate similarity)."""
        return -self.tversky_sim(a, b)

    def save_model(self, path):
        """Save state_dict + constructor hparams so load_model can reconstruct exactly."""
        torch.save({
            'hparams': {
                'input_dim': self.encoder[0].in_features,
                'hidden_dim': self.encoder[0].out_features,
                'latent_dim': self.encoder[-1].out_features,
                'fbank_size': self.tversky_sim.feature_bank.weight.shape[0],
                'similarity_model': self.tversky_sim.similarity_model,
                'normalize': self.tversky_sim.normalize,
                'intersection_reduction': self.tversky_sim.intersection_reduction,
                'difference_reduction': self.tversky_sim.difference_reduction,
            },
            'state_dict': self.state_dict(),
        }, path)
        print(f"model saved to {path}")


def symmetric_triplet_loss(model, a_emb, p_emb, n_emb, margin=1.0):
    """L_trip(A, P, N) + L_trip(P, A, N), using model's Tversky distance."""
    loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=model.distance, margin=margin, reduction='mean'
    )
    return loss_fn(a_emb, p_emb, n_emb) + loss_fn(p_emb, a_emb, n_emb)


def asymmetric_triplet_loss(model, a_emb, p_emb, n_emb, margin=1.0):
    loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=model.distance, margin=margin, reduction='mean'
    )
    return loss_fn(a_emb, p_emb, n_emb)


def train_tversky_sirl(
    anchors, positives, negatives,
    num_epochs=3000,
    batch_size=64,
    lr=0.004,
    lr_decay=0.99999,
    margin=1.0,
    latent_dim=6,
    hidden_dim=1024,
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

    input_dim = A.shape[1] * A.shape[2]
    n_triplets = A.shape[0]

    model = TverskySIRL(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
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
                ae, pe, ne = model(A), model(P), model(N)
                # Eval accuracy uses the same Tversky distance
                d_ap = model.distance(ae, pe)
                d_an = model.distance(ae, ne)
                d_pn = model.distance(pe, ne)
                acc_a = (d_an > d_ap).float().mean().item()
                acc_p = (d_pn > d_ap).float().mean().item()
                acc = 0.5 * (acc_a + acc_p)
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} | "
                  f"triplet_acc={acc:.3f} | lr={scheduler.get_last_lr()[0]:.5f}")
            history.append({'epoch': epoch, 'loss': loss.item(), 'acc': acc})

    return model, history

def load_model(path, device=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TverskySIRL(**ckpt['hparams'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    if device:
        model = model.to(device)
    return model
