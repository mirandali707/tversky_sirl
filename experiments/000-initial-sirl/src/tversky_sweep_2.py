from utils import *
from tversky_sirl_2 import *
from fpe import *
import numpy as np
import torch
import random
import json
sys.path.insert(0, os.path.abspath('..'))
from src.utils.input_utils import transform_input

# --- Sweep config ---
N_QUERIES = 1000
# FBANK_SIZES = [4, 16, 64, 256]
FBANK_SIZES = [512, 1024, 2048]
OUTFILE_TEMPLATE = "061826_tversky_2_{fbank}_results.json"

# JacoRobot input transform: drops rotation-matrix (63) + joint-angle (7) dims,
# leaving the 27 xyz position dims per state. (N,21,97) -> (N,21,27).
# Both features (table, laptop) are positional, so the dropped dims are distractors.
INPUT_DICT = {'lowdim': False, 'EErot': False, 'noxyz': False, 'norot': True, 'noangles': True}


def tx(trajs):
    """Apply the JacoRobot input transform to a (N, 21, 97) trajectory array."""
    x = torch.as_tensor(trajs, dtype=torch.float32)
    return transform_input(x, INPUT_DICT).numpy()


# --- Seeds ---
# Fixed across the whole experiment: ensures the FPE eval set is identical
# for every SIRL model we evaluate. NEVER change this between conditions.
FPE_SPLIT_SEED = 42

# SIRL training seeds: vary across these to get error bars on FPE.
SIRL_SEEDS = [0, 1, 2, 3, 4]


def set_all_seeds(seed):
    """Set seeds for every source of randomness in SIRL training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Build FPE eval data ONCE, outside all loops ---
# Uses FPE_SPLIT_SEED so the split is deterministic and shared.
train_trajs, train_features, test_trajs, test_features = \
    get_train_test_trajs_feats(seed=FPE_SPLIT_SEED)
train_trajs = tx(train_trajs)
test_trajs = tx(test_trajs)


# --- Load triplet data ONCE (same for all fbank sizes) ---
print(f"Loading {N_QUERIES} queries...")
anchors, positives, negatives = load_data(N_QUERIES)
anchors, positives, negatives = tx(anchors), tx(positives), tx(negatives)


# --- Sweep over feature bank sizes ---
all_results = {}
for fbank_size in FBANK_SIZES:
    outfile = OUTFILE_TEMPLATE.format(fbank=fbank_size)
    print(f"\n=== TRAINING TVERSKY 2 SIRL | n_queries={N_QUERIES} | fbank_size={fbank_size} ===")

    fpes_for_fbank = []
    for seed in SIRL_SEEDS:
        print(f"  -- seed {seed} --")
        set_all_seeds(seed)

        model, _ = train_tversky_sirl(
            anchors, positives, negatives,
            use_symmetric_loss=True,
            fbank_size=fbank_size,
        )

        fpe = eval_fpe(model, train_trajs, train_features, test_trajs, test_features)

        print(f"     FPE = {fpe:.4f}")
        fpes_for_fbank.append(fpe)

    fpes = np.array(fpes_for_fbank)
    results = {
        "n_queries": N_QUERIES,
        "fbank_size": fbank_size,
        "mean": float(fpes.mean()),
        "std":  float(fpes.std(ddof=1)),
        "all":  fpes.tolist(),
        "seeds": SIRL_SEEDS,
    }
    print(f"  fbank={fbank_size}: FPE = {fpes.mean():.4f} ± {fpes.std(ddof=1):.4f}")

    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  wrote: {outfile}")

    all_results[fbank_size] = results

print("\nDone.")
print(json.dumps(all_results, indent=2))