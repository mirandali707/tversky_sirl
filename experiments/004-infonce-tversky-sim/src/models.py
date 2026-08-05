import time
import numpy as np
from literally_just_tversky_sim import train_tversky_sim, random_init_no_training_tversky_sim
from ljts_triplet import train_triplet_tversky_sim
from encoders import *

def train_model(config, data, results_dir, seed):
    """
    NOTE we are only training on the laptop feature -
    from all trajectories and their feature values, we extract only trajectories with max or min laptop distance
    and train just the TverskySimilarity layer with InfoNCE loss on those
    """
    unix_timestamp = int(time.time())

    if config["experiment_name"] == "random_init_no_training":
        # TODO this only uses input dim 19, could edit to take different input dims for comparison with e.g. pca, sirl
        model = random_init_no_training_tversky_sim(config)
        ckpt_path = str(results_dir / f"random_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
        return model, ckpt_path

    if config["experiment_name"] == "ljts_triplet":
        anchors = data["anchors"]
        positives = data["positives"]
        negatives = data["negatives"]

        model = train_triplet_tversky_sim(config, anchors, positives, negatives)
        ckpt_path = str(results_dir / f"random_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
        return model, ckpt_path

    hi_trajs, lo_trajs, hi_feats, lo_feats = get_hi_lo_laptop_trajs(data)
    input_dim = hi_trajs.shape[1]
    # transform trajs with encoder, if specified
    if config["model"]["encoder"] == "pca":
        # use PCA embeds as input
        latent_dim = config["model"]["latent_dim"]
        hi_trajs, input_dim = pca(hi_trajs, latent_dim)
        lo_trajs, input_dim = pca(lo_trajs, latent_dim)
    if config["model"]["encoder"] == "sirl":
        # use SIRL embeds as input
        latent_dim = config["model"]["latent_dim"]
        hi_trajs, input_dim = sirl(hi_trajs, latent_dim)
        lo_trajs, input_dim = sirl(lo_trajs, latent_dim)
    if config["model"]["encoder"] == "feats":
        # use ground truth features as input
        hi_trajs = hi_feats
        lo_trajs = lo_feats
        input_dim = hi_feats.shape[1]
    model = train_tversky_sim(config, hi_trajs, lo_trajs, input_dim=input_dim)
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
    hi_feats = all_feats[laptop_max_indices]
    lo_feats = all_feats[laptop_min_indices]
    print(f"{len(hi_trajs)} laptop max trajs and {len(lo_trajs)} laptop min trajs")
    return hi_trajs, lo_trajs, hi_feats, lo_feats