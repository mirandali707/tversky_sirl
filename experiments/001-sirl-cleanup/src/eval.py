import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression


def eval_model(config, data, model):
    """
    eval
    """
    print("eval")
    eval_params = config["eval"]
    if isinstance(eval_params, dict):  # single method given as a bare dict
        eval_params = [eval_params]
    all_eval = {}
    for entry in eval_params:
        method = entry["method"]
        if method == "fpe":
            fpe = eval_fpe(config, data, model)
            print(f"fpe: {fpe}")
            all_eval = all_eval | {"fpe": fpe}
        if method == "tpa":
            tpa = eval_tpa(config, data, model)
            print(f"tpa: {tpa}")
            all_eval = all_eval | {"tpa": tpa}
    return all_eval


def eval_fpe(config, data, model):
    train_trajs = data["train_trajs"]
    test_trajs = data["test_trajs"]
    train_features = data["train_features"]
    test_features = data["test_features"]

    # get embeddings (run train, test data through model)
    if config["model"]["name"] == "pca":
        # sklearn pca
        Z_train = model.transform(train_trajs.reshape(len(train_trajs), -1))
        Z_test  = model.transform(test_trajs.reshape(len(test_trajs), -1))
    else:
        # pytorch (e.g. SIRL)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        with torch.no_grad():
            Z_train = model(torch.as_tensor(train_trajs, dtype=torch.float32, device=device)).cpu().numpy()
            Z_test  = model(torch.as_tensor(test_trajs,  dtype=torch.float32, device=device)).cpu().numpy()
    # fit fpe probe on train embeds -> ground truth features
    reg = LinearRegression().fit(Z_train, train_features)
    pred = reg.predict(Z_test) # predict test labels
    fpe = np.mean(np.sum((pred - test_features) ** 2, axis=1)) # report mse
    return fpe


def eval_tpa(config, data, model):
    pref_pairs  = data["pref_pairs"]   # (n_pref, 2) - indices into trajs
    pref_labels = data["pref_labels"]  # (n_pref,)  - 0 if traj[pair[0]] preferred, else 1
    trajs       = data["trajs"]        # (n_traj, ...) - trajectory pool the pairs index into

    # get embeddings (run all trajs through model) -- same branch as eval_fpe
    if config["model"]["name"] == "pca":
        Z = model.transform(trajs.reshape(len(trajs), -1))
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        with torch.no_grad():
            Z = model(torch.as_tensor(trajs, dtype=torch.float32, device=device)).cpu().numpy()

    # train/test split over PREFERENCE PAIRS (the split that must be clean for tpa)
    rng = np.random.default_rng(config.get("seed", 0))
    n = len(pref_pairs)
    perm = rng.permutation(n)
    n_test = int(round(config.get("tpa_test_frac", 0.2) * n))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train_pairs,  test_pairs  = pref_pairs[train_idx],  pref_pairs[test_idx]
    train_labels, test_labels = pref_labels[train_idx], pref_labels[test_idx]

    # torch tensors for the reward head (embeddings are frozen, like FPE's probe)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Z = torch.as_tensor(Z, dtype=torch.float32, device=device)
    train_pairs  = torch.as_tensor(train_pairs,  dtype=torch.long,    device=device)
    test_pairs   = torch.as_tensor(test_pairs,   dtype=torch.long,    device=device)
    train_labels = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    test_labels  = torch.as_tensor(test_labels,  dtype=torch.long,    device=device)

    # reward head R_theta on top of frozen embeddings
    hidden, n_layers = config.get("tpa_hidden", 128), config.get("tpa_layers", 2)
    layers, in_dim = [], Z.shape[1]
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden), nn.ReLU()]; in_dim = hidden
    layers += [nn.Linear(in_dim, 1)]
    reward = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(reward.parameters(), lr=config.get("tpa_lr", 1e-3),
                           weight_decay=config.get("tpa_l2", 0.0))
    bce = nn.BCEWithLogitsLoss()

    # Bradley-Terry: P(A>B) = sigmoid(R(A) - R(B))  (Eq. 3); BCE on logits = Eq. (4)
    # label 0 => A preferred => target prob(A>B) = 1; label 1 => target = 0  => y = 1 - label
    A_tr, B_tr = train_pairs[:, 0], train_pairs[:, 1]
    y_tr = 1.0 - train_labels
    batch_size = config.get("tpa_batch_size", 64)
    reward.train()
    for _ in range(config.get("tpa_epochs", 500)):
        bperm = torch.randperm(len(train_pairs), device=device)
        for s in range(0, len(bperm), batch_size):
            bi = bperm[s:s + batch_size]
            logits = reward(Z[A_tr[bi]]).squeeze(-1) - reward(Z[B_tr[bi]]).squeeze(-1)
            loss = bce(logits, y_tr[bi])
            opt.zero_grad(); loss.backward(); opt.step()

    # TPA = preference accuracy on held-out pairs
    reward.eval()
    with torch.no_grad():
        rA = reward(Z[test_pairs[:, 0]]).squeeze(-1)
        rB = reward(Z[test_pairs[:, 1]]).squeeze(-1)
        pred = (rA <= rB).long()                 # rA>rB -> A preferred -> label 0
        tpa = (pred == test_labels).float().mean().item()

    return tpa