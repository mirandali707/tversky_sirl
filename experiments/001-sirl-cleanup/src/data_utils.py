import numpy as np
from pathlib import Path

# Repository root, used to resolve data paths in configs (which are relative to it).
# data_utils.py lives at <repo>/experiments/001-sirl-cleanup/src/data_utils.py
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_data(config):
    """
    loads simulated data from .npz
    data object has "anchors", "positives", "negatives" keys
    each is a np array with shape (n_triplets, 19) since 19 is gridrobot trajectory dim (9 xy coords + one joint angle)
    returns anchors, positives, negatives
    """
    data_params = config["data"]
    print(f"loading data: {data_params['dataset_name']}")

    path = REPO_ROOT / data_params["path"]
    npz = np.load(path)
    return (npz["anchors"], npz["positives"], npz["negatives"])
