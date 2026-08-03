import time
import numpy as np
from literally_just_tversky_sim import train_tversky_sim

def train_model(config, data, results_dir, seed):
    """
    NOTE we are only training on the laptop feature -
    from all trajectories and their feature values, we extract only trajectories with max or min laptop distance
    and train just the TverskySimilarity layer with InfoNCE loss on those
    """
    hi_trajs, lo_trajs = get_hi_lo_laptop_trajs(data)

    model = train_tversky_sim(config, hi_trajs, lo_trajs)
    unix_timestamp = int(time.time())
    ckpt_path = str(results_dir / f"tversky_proj_{unix_timestamp}.pth")
    model.save_model(ckpt_path)
    return model, ckpt_path

def get_hi_lo_laptop_trajs(data):
    all_trajs = data["trajs"]
    all_feats = data["features"]

    laptop_dist = all_feats[:,0]
    laptop_max_indices = np.where(laptop_dist == laptop_dist.max())
    laptop_min_indices = np.where(laptop_dist == laptop_dist.min())
    hi_trajs = all_trajs[laptop_max_indices]
    lo_trajs = all_trajs[laptop_min_indices]
    print(f"{len(hi_trajs)} laptop max trajs and {len(lo_trajs)} laptop min trajs")
    return hi_trajs, lo_trajs