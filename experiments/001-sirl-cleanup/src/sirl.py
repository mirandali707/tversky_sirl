# simplified version of src/models/featurizers/similarity.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from utils import config_overridable


class SIRL(nn.Module):
    """
    Minimal SIRL: MLP encoder trained with symmetric triplet margin loss
    on (anchor, positive, negative) trajectory triplets.

    JacoRobot architecture (from paper Appendix A.2.1):
      - 2 hidden layers of 1024 units, ReLU
      - 6-dim latent output
      - Input: flattened trajectory (21 * 97 = 2037)
    """
    def __init__(self, input_dim=2037, hidden_dim=1024, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        z = self.encoder(x)
        return z


def symmetric_triplet_loss(anchor_emb, pos_emb, neg_emb, margin=1.0):
    """
    Symmetric triplet loss (SIRL Eq. 2):
        L = L_trip(P1, P2, N) + L_trip(P2, P1, N)

    Since the original similarity query has no explicit anchor (just
    "which two are most similar"), both similar trajectories take turns
    as the anchor.
    """
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    loss_ap = loss_fn(anchor_emb, pos_emb, neg_emb)   # anchor as anchor
    loss_pa = loss_fn(pos_emb, anchor_emb, neg_emb)   # positive as anchor
    return 30 * (loss_ap + loss_pa)


def asymmetric_triplet_loss(anchor_emb, pos_emb, neg_emb, margin=1.0):
    """
    normal triplet loss, which is what Similarity (in similarity.py) does...
    """
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    loss_ap = loss_fn(anchor_emb, pos_emb, neg_emb)
    return 30 * (loss_ap)


@config_overridable
def train_sirl(
    config,
    anchors, positives, negatives,
    num_epochs=3000,
    batch_size=64,
    lr=0.004,
    lr_decay=0.99999,
    margin=0.1,
    latent_dim=6,
    hidden_dim=1024,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    use_symmetric_loss=True,
    l2_weight=0.0,
):
    """
    train SIRL on labelled triplets.
    anchors, positives, negatives: np.ndarray of shape (N, ...) where ... is observation dim (19 for gridrobot, (21, 97) for jaco)
    where anchors are more similar to positives than negatives
    """
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

    model = SIRL(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
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
            # Eqn 2 in SIRL paper
            loss = symmetric_triplet_loss(a_emb, p_emb, n_emb, margin=margin)
        else:
            # implementation in similarity.py
            loss = asymmetric_triplet_loss(a_emb, p_emb, n_emb, margin=margin)

        if l2_weight > 0.0:
            l2_fn = nn.MSELoss()
            for emb in (a_emb, p_emb, n_emb):
                loss = loss + l2_weight * l2_fn(emb, torch.zeros_like(emb))

        # NOTE not handling skipped queries...

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
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


# ---------------- Run ----------------
# if __name__ == "__main__":
#     with np.load('sim_100_0.npz') as data:
#         anchors = data['anchors']      # (100, 21, 97)
#         positives = data['positives']
#         negatives = data['negatives']

#     model, history = train_sirl(anchors, positives, negatives)