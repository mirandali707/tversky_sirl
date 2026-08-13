import torch
import wandb
from ljts_triplet import train_triplet_tversky_sim, random_init_no_training_tversky_sim
from sirl_ts import train_sirl_tversky_sim
from encoders import get_sirl

def train_model(config, data, results_dir, seed, unix_timestamp):
    """
    NOTE we are only training on the laptop feature -
    from all trajectories and their feature values, we extract only trajectories with max or min laptop distance
    and train just the TverskySimilarity layer with InfoNCE loss on those
    """
    expt_name = config["experiment_name"]

    if expt_name == "random_no_sirl_ts":
        model = random_init_no_training_tversky_sim(config)
        ckpt_path = str(results_dir / f"{expt_name}_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
        return model, ckpt_path, "no_run"

    # get (a, p, n) data
    anchors = data["anchors"]
    positives = data["positives"]
    negatives = data["negatives"]

    # init wandb run
    run = wandb.init(
        entity="m1randa-massachusetts-institute-of-technology",
        project="tversky-sirl",
        config=config,
    )
    run_url = wandb.run.url

    if expt_name == "no_sirl_ts" or expt_name == "tiny":
        model = train_triplet_tversky_sim(config, anchors, positives, negatives, wandb_run=run)
        ckpt_path = str(results_dir / f"{expt_name}_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
        run.finish()
        return model, ckpt_path, run_url

    if expt_name == "frozen_sirl_ts":
        # use SIRL embeds as input (transform a, p, n trajectories)
        latent_dim = config["model"]["latent_dim"]
        encoder = get_sirl(seed, latent_dim) # loads trained sirl checkpoint with no_grad_(True)
        anchors = encoder(torch.tensor(anchors, dtype=torch.float32)).detach()
        positives = encoder(torch.tensor(positives, dtype=torch.float32)).detach()
        negatives = encoder(torch.tensor(negatives, dtype=torch.float32)).detach()

        model = train_triplet_tversky_sim(config, anchors, positives, negatives, input_dim=latent_dim, wandb_run=run)
        ckpt_path = str(results_dir / f"{expt_name}_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
        run.finish()
        return model, ckpt_path, run_url

    if expt_name == "sirl_ts":
        model = train_sirl_tversky_sim(config, anchors, positives, negatives, wandb_run=run)
        ckpt_path = str(results_dir / f"{expt_name}_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
        run.finish()
        return model, ckpt_path, run_url

    
    run.finish() # we should never get here but just in case
    return None, None, None