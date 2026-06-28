import numpy as np
import random
import torch
from pathlib import Path
import functools
import inspect

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


def config_overridable(fn):
    """
    override fn params with matching keys from config['model'].
    used for fns that train models, e.g. overriding n_components in models.py:fit_pca
    """
    sig = inspect.signature(fn)
    @functools.wraps(fn)
    def wrapper(config, *args, **kwargs):
        model_cfg = (config or {}).get("model", {})
        for name, param in sig.parameters.items():
            # only override real keyword params, and only if the caller didn't pass it explicitly
            if (param.default is not inspect.Parameter.empty
                    and name in model_cfg
                    and name not in kwargs):
                kwargs[name] = model_cfg[name]
        return fn(config, *args, **kwargs)
    return wrapper