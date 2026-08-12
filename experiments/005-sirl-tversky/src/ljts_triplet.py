from tversky import nn as tnn
from info_nce import InfoNCE, info_nce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from utils import config_overridable


class LiterallyJustTverskySim(nn.Module):
    def __init__(self, 
                 input_dim=19, 
                 fbank_size=4, 
                 similarity_model='contrast',
                 intersection_reduction='product',
                 difference_reduction='ignorematch',
                 normalize=False
                 ):
        super().__init__()
        self.tversky_sim = tnn.TverskySimilarity(
            embedding_dim=input_dim,
            fbank_size=fbank_size,
            similarity_model=similarity_model,
            normalize=normalize,
            intersection_reduction=intersection_reduction,
            difference_reduction=difference_reduction
        )
        torch.nn.init.uniform_(self.tversky_sim.feature_bank.weight, -1.0, 1.0)

        # store constructor args so save_model/load can reconstruct exactly
        self.hparams = {
            'input_dim': input_dim,
            'fbank_size': fbank_size,
            'similarity_model': similarity_model,
            'intersection_reduction': intersection_reduction,
            'difference_reduction': difference_reduction,
            'normalize': normalize,
        }
    
    def similarity(self, a, b):
        return self.tversky_sim(a,b)

    def save_model(self, path):
        """Save state_dict + constructor hparams so load_model can reconstruct exactly."""
        torch.save({
            'hparams': self.hparams,
            'state_dict': self.state_dict(),
        }, path)
        print(f"model saved to {path}")
    

@config_overridable
def train_triplet_tversky_sim(
    config,
    anchors, positives, negatives,
    margin=0.1,
    symmetric=True,
    input_dim=19,
    fbank_size=4, 
    similarity_model='contrast',
    intersection_reduction='product',
    difference_reduction='ignorematch',
    normalize=False,
    num_epochs=10000,
    batch_size=64,
    lr=0.004,
    lr_decay=0.99999,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
    wandb_run=None
):

    model = LiterallyJustTverskySim(
        input_dim=input_dim,
        fbank_size=fbank_size, 
        similarity_model=similarity_model,
        intersection_reduction=intersection_reduction,
        difference_reduction=difference_reduction,
    ).to(device)
    # print("PARAMS")
    # print(dict(model.named_parameters()).keys())

    # Adam now optimizes encoder + Tversky feature bank + α/β/θ
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    A = torch.as_tensor(anchors, dtype=torch.float32, device=device).flatten(start_dim=1)
    P = torch.as_tensor(positives, dtype=torch.float32, device=device).flatten(start_dim=1)
    N = torch.as_tensor(negatives, dtype=torch.float32, device=device).flatten(start_dim=1)

    assert A.shape == P.shape == N.shape, f"shape mismatch: {A.shape}, {P.shape}, {N.shape}"
    n_triplets = A.shape[0]

    for epoch in range(num_epochs):
        model.train()
        idx = np.random.choice(n_triplets, size=min(batch_size, n_triplets), replace=False)
        idx = torch.as_tensor(idx, device=device)
        a, p, n = A[idx], P[idx], N[idx]

        if symmetric:
            loss = symmetric_triplet_loss(model, a, p, n, margin)
        else:
            loss = asymmetric_triplet_loss(model, a, p, n, margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                s_ap = model.similarity(A, P)
                s_an = model.similarity(A, N)
                acc = (s_ap > s_an).float().mean().item()
                curr_lr = scheduler.get_last_lr()[0]
                mean_s_ap = s_ap.mean()
                std_s_ap = s_ap.std()
                mean_s_an = s_an.mean()
                std_s_an = s_an.std()
                mean_gap_sap_san = (s_ap - s_an).mean()
                if wandb_run:
                    wandb_run.log({
                        "epoch": epoch, 
                        "acc": acc, 
                        "loss": loss, 
                        "lr": curr_lr,
                        "mean_s_ap": mean_s_ap,
                        "std_s_ap": std_s_ap,
                        "mean_s_an": mean_s_an,
                        "std_s_an": std_s_an,
                        "mean_gap_sap_san": mean_gap_sap_san
                        })
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} | "
                  f"triplet_acc={acc:.3f} | lr={curr_lr:.5f}")
            # collapse diagnostic: if s_pos ~= s_neg (mean gap ~0, small std),
            # the model has collapsed to a near-constant similarity and the
            # loss will floor at 2*margin (symmetric) / margin (asymmetric).
            print(f"           | s_ap: mean={mean_s_ap:.4f} std={std_s_ap:.4f} | "
                  f"s_an: mean={mean_s_an:.4f} std={std_s_an:.4f} | "
                  f"gap(mean s_ap-s_an)={mean_gap_sap_san:.4f}")

    return model

@config_overridable
def random_init_no_training_tversky_sim(
    config,
    input_dim=19,
    fbank_size=4, 
    similarity_model='contrast',
    intersection_reduction='product',
    difference_reduction='ignorematch',
    normalize=False,
    num_epochs=10000,
    batch_size=64,
    lr=0.004,
    lr_decay=0.99999,
    tau=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100,
):
    model = LiterallyJustTverskySim(
        input_dim=input_dim,
        fbank_size=fbank_size, 
        similarity_model=similarity_model,
        intersection_reduction=intersection_reduction,
        difference_reduction=difference_reduction,
    ).to(device)
    return model


def _margin_ranking_loss(s_pos, s_neg, margin):
    """
    Triplet margin loss on *similarities* (not distances):
    encourage sim(anchor, positive) to exceed sim(anchor, negative) by `margin`.
        L = mean( relu(margin - s_pos + s_neg) )
    """
    return F.relu(margin - s_pos + s_neg).mean()


def symmetric_triplet_loss(model, anchor, pos, neg, margin=1.0):
    """
    Symmetric triplet loss (SIRL Eq. 2) adapted for a similarity model:
        L = L_trip(A, P, N) + L_trip(P, A, N)

    Since the original similarity query has no explicit anchor (just
    "which two are most similar"), both similar trajectories take turns
    as the anchor. TverskySimilarity is itself asymmetric, so the two
    terms are genuinely different.
    """
    loss_ap = _margin_ranking_loss(model.similarity(anchor, pos),
                                   model.similarity(anchor, neg), margin)
    loss_pa = _margin_ranking_loss(model.similarity(pos, anchor),
                                   model.similarity(pos, neg), margin)
    return loss_ap + loss_pa


def asymmetric_triplet_loss(model, anchor, pos, neg, margin=1.0):
    """
    normal triplet loss, which is what Similarity (in similarity.py) does...
    """
    loss_ap = _margin_ranking_loss(model.similarity(anchor, pos),
                                   model.similarity(anchor, neg), margin)
    return loss_ap


def load_tversky_sim(path, device=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LiterallyJustTverskySim(**ckpt['hparams'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    if device:
        model = model.to(device)
    return model
