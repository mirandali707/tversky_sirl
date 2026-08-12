import time
import numpy as np
import wandb
from ljts_triplet import train_triplet_tversky_sim

def train_model(config, data, results_dir, seed):
    """
    NOTE we are only training on the laptop feature -
    from all trajectories and their feature values, we extract only trajectories with max or min laptop distance
    and train just the TverskySimilarity layer with InfoNCE loss on those
    """
    unix_timestamp = int(time.time())
    expt_name = config["experiment_name"]
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="m1randa-massachusetts-institute-of-technology",
        # Set the wandb project where this run will be logged.
        project="tversky-sirl",
        # Track hyperparameters and run metadata.
        config=config,
    )

    if expt_name == "no_sirl_ts":
        # TODO this only uses input dim 19, could edit to take different input dims for comparison with e.g. pca, sirl
        anchors = data["anchors"]
        positives = data["positives"]
        negatives = data["negatives"]

        model = train_triplet_tversky_sim(config, anchors, positives, negatives, wandb_run)
        ckpt_path = str(results_dir / f"{expt_name}_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
        
        run.finish()
        return model, ckpt_path

    # if config["experiment_name"] == "ljts_triplet":

    # hi_trajs, lo_trajs, hi_feats, lo_feats = get_hi_lo_laptop_trajs(data)
    # input_dim = hi_trajs.shape[1]
    # # transform trajs with encoder, if specified
    # if config["model"]["encoder"] == "pca":
    #     # use PCA embeds as input
    #     latent_dim = config["model"]["latent_dim"]
    #     hi_trajs, input_dim = pca(hi_trajs, latent_dim)
    #     lo_trajs, input_dim = pca(lo_trajs, latent_dim)
    # if config["model"]["encoder"] == "sirl":
    #     # use SIRL embeds as input
    #     latent_dim = config["model"]["latent_dim"]
    #     hi_trajs, input_dim = sirl(hi_trajs, latent_dim)
    #     lo_trajs, input_dim = sirl(lo_trajs, latent_dim)
    # if config["model"]["encoder"] == "feats":
    #     # use ground truth features as input
    #     hi_trajs = hi_feats
    #     lo_trajs = lo_feats
    #     input_dim = hi_feats.shape[1]
    # model = train_tversky_sim(config, hi_trajs, lo_trajs, input_dim=input_dim)
    # ckpt_path = str(results_dir / f"tversky_proj_{unix_timestamp}.pth")
    # model.save_model(ckpt_path)
    # return model, ckpt_path
