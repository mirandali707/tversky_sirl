from utils import *
from tversky_sirl import *
from fpe import *
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from src.utils.input_utils import transform_input
import numpy as np
import torch
import random
import json

OUTFILE = "061826_pca_results.json"

# --- Seeds ---
# Fixed across the whole experiment: ensures the FPE eval set is identical
# for every SIRL model we evaluate. NEVER change this between conditions.
FPE_SPLIT_SEED = 42

# SIRL training seeds: vary across these to get error bars on FPE.
SIRL_SEEDS = [0, 1, 2, 3, 4]
n_queries = 1000

# JacoRobot input transform: drops rotation-matrix (63) + joint-angle (7) dims,
# leaving the 27 xyz position dims per state. (N,21,97) -> (N,21,27).
# Both features (table, laptop) are positional, so the dropped dims are distractors.
INPUT_DICT = {'lowdim': False, 'EErot': False, 'noxyz': False, 'norot': True, 'noangles': True}


def tx(trajs):
    """Apply the JacoRobot input transform to a (N, 21, 97) trajectory array."""
    x = torch.as_tensor(trajs, dtype=torch.float32)
    return transform_input(x, INPUT_DICT).numpy()


def set_all_seeds(seed):
    """Set seeds for every source of randomness in SIRL training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Build FPE eval data ONCE, outside the seed loop ---
# Uses FPE_SPLIT_SEED so the split is deterministic and shared.
train_trajs, train_features, test_trajs, test_features = \
    get_train_test_trajs_feats(seed=FPE_SPLIT_SEED)

# Transform the eval trajectories (features are unchanged — they're labels).
train_trajs = tx(train_trajs)
test_trajs = tx(test_trajs)


results = {}
print(f"\n=== TRAINING SIRL FOR {n_queries} QUERIES ===")
anchors, positives, negatives = load_data(n_queries)

# Transform the training triplets identically.
anchors, positives, negatives = tx(anchors), tx(positives), tx(negatives)

fpes_for_n = []
for seed in SIRL_SEEDS:
    print(f"  -- seed {seed} --")
    set_all_seeds(seed)

    query_trajs = np.concatenate([anchors, positives, negatives], axis=0)
    pca = fit_pca(query_trajs, n_components=6, random_state=seed)

    fpe = eval_pca_fpe(pca, train_trajs, train_features, test_trajs, test_features)

    # fpe, probe = eval_fpe_probe(model, train_trajs, train_features, test_trajs, test_features)
    fpe = eval_fpe(model, train_trajs, train_features, test_trajs, test_features)

    # print(f"     FPE = {fpe:.4f}")
    print(f"     FPE = {fpe:.4f}")
    Z = model(torch.as_tensor(test_trajs, dtype=torch.float32, device='cpu')).detach().numpy()
    print(np.linalg.norm(Z, axis=1).mean(), np.linalg.cond(Z))
    fpes_for_n.append(fpe)

fpes = np.array(fpes_for_n)
results[n_queries] = {
    "mean": float(fpes.mean()),
    "std":  float(fpes.std(ddof=1)),
    "all":  fpes.tolist(),
    "seeds": SIRL_SEEDS,
}
print(f"  N={n_queries}: FPE = {fpes.mean():.4f} ± {fpes.std(ddof=1):.4f}")

with open(OUTFILE, "w") as f:
    json.dump(results, f, indent=4)