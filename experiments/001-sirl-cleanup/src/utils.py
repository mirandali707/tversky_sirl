import numpy as np
import random
import torch
from pathlib import Path

# Repository root, used to resolve data paths in configs (which are relative to it).
# data_utils.py lives at <repo>/experiments/001-sirl-cleanup/src/data_utils.py
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_data(config):
    """
    loads simulated data from .npz
    """
    data_params = config["data"]
    print(f"loading data: {data_params['dataset_name']}")

    path = REPO_ROOT / data_params["path"]
    return np.load(path)


def set_all_seeds(seed):
    """Set seeds for every source of randomness in SIRL training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)