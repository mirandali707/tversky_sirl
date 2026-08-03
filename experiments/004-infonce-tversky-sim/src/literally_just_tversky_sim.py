from tversky import nn as tnn
from info_nce import InfoNCE, info_nce
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
def train_tversky_sim(
    config,
    hi_trajs, lo_trajs,
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

    # Adam now optimizes encoder + Tversky feature bank + α/β/θ
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    hi = torch.as_tensor(hi_trajs, dtype=torch.float32)   # (N_hi, d)
    lo = torch.as_tensor(lo_trajs, dtype=torch.float32)   # (N_lo, d)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = contrastive_step(model, hi, lo, batch_size, tau)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % log_interval == 0:
            print(loss.item())
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.5f}")

    return model


def contrastive_step(model, hi, lo, batch_size, tau=0.1):
    # anchors: batch_size positives; partners: also batch_size positives
    a_idx = torch.randperm(len(hi))[:batch_size]
    p_idx = torch.randperm(len(hi))[:batch_size]          # different draw -> different positive
    anchors  = hi[a_idx]                          # (batch_size, d)
    partners = hi[p_idx]                          # (batch_size, d)
    negs     = lo[torch.randperm(len(lo))[:batch_size]]    # (batch_size, d)  the negative pool

    # pairwise sim returns (rows, cols)
    s_pos = model.similarity(anchors, partners).diagonal().unsqueeze(1)  # (batch_size, 1): anchor i vs partner i
    s_neg = model.similarity(anchors, negs)                              # (batch_size, batch_size): anchor i vs every neg

    logits = torch.cat([s_pos, s_neg], dim=1) / tau   # (batch_size, 1+batch_size)
    labels = torch.zeros(batch_size, dtype=torch.long)          # positive is column 0
    return F.cross_entropy(logits, labels)


def load_tversky_sirl(path, device=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LiterallyJustTverskySim(**ckpt['hparams'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    if device:
        model = model.to(device)
    return model
