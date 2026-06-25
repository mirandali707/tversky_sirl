# simplified version of src/models/evaluation/feature_learner.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.optim as optim
from src.models.mlp import MLP


class FPEProbe:
    """
    MLP probe for Feature Prediction Error (FPE).

    Freezes a trained SIRL model and fits a small MLP on top
    to predict ground-truth features. FPE = MSE between predicted and
    true features on a held-out test set.

    Lower FPE => the embedding better captures the human's true features.
    """
    def __init__(self, sirl_model, latent_dim=6, num_features=4,
                 hiddens=[64, 64],
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = sirl_model.to(device).eval()       # frozen featurizer
        self.probe = MLP(latent_dim, num_features, []).to(device)

    def _embed(self, trajs):
        """Get frozen embeddings for a batch of trajectories."""
        with torch.no_grad():
            x = torch.as_tensor(trajs, dtype=torch.float32, device=self.device)
            return self.model(x)

    def _loss(self, trajs, features, batch_size):
        idx = np.random.choice(len(trajs),
                               size=min(batch_size, len(trajs)),
                               replace=False)
        emb = self._embed(trajs[idx])
        feat = torch.as_tensor(features[idx], dtype=torch.float32,
                               device=self.device)
        pred = self.probe(emb)
        return torch.mean(torch.sum((pred - feat) ** 2, dim=1))

    def train(self, trajs, features, test_trajs=None, test_features=None,
              num_iters=20000, lr=.001, batch_size=64, # from configs/evaluator/feature_learner.yaml
              log_interval=200):
        """Fit the MLP probe by gradient descent."""
        self.last_train_loss = None
        self.last_test_loss = None

        opt = optim.Adam(self.probe.parameters(), lr=lr)
        for i in range(num_iters):
            loss = self._loss(trajs, features, batch_size)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % log_interval == 0:
                print(f"  probe iter {i:4d} | train_loss={loss.item():.4f}")

        with torch.no_grad():
            for _ in range(100):
                train_loss = self._loss(trajs, features, batch_size)
                if self.last_train_loss is None:
                    self.last_train_loss = train_loss.item()
                self.last_train_loss = self.last_train_loss * 0.9 + train_loss.item() * 0.1

                if test_trajs is not None and test_features is not None:
                    test_loss = self._loss(test_trajs, test_features, batch_size)
                    if self.last_test_loss is None:
                        self.last_test_loss = test_loss.item()
                    self.last_test_loss = self.last_test_loss * 0.9 + test_loss.item() * 0.1

        return self

    def evaluate(self, trajs, features):
        """Compute FPE (mean squared error) over the full set."""
        with torch.no_grad():
            emb = self._embed(trajs)
            feat = torch.as_tensor(features, dtype=torch.float32,
                                   device=self.device)
            pred = self.probe(emb)
            fpe = torch.mean(torch.sum((pred - feat) ** 2, dim=1)).item()
        return fpe


# ---------------- Usage ----------------
# if __name__ == "__main__":
#     # `model` is your trained SIRL model
#     # `train_trajs`, `test_trajs`: np.ndarray of shape (N, 21, 97)
#     # `train_features`, `test_features`: np.ndarray of shape (N, 4)
#     #   computed from the simulated human's ground-truth feature function

#     probe = FPEProbe(model, latent_dim=6, num_features=2)
#     probe.train(train_trajs, train_features, num_iters=2000)
#     fpe = probe.evaluate(test_trajs, test_features)
#     print(f"FPE = {fpe:.4f}")