from pca import *
from sirl import train_sirl, load_sirl, init_random_sirl
from tversky_sim import train_tversky_sirl, load_tversky_sirl
from tversky_proj_sim import train_tversky_sirl_2, load_tversky_sirl_2
from tversky_proj import train_tversky_proj, load_tversky_proj
from tversky_vae import train_tversky_vae, load_tversky_vae
import time


def load_model(train_config, ckpt_path):
    """
    loads a saved model checkpoint.
    """
    model_params = train_config["model"]
    if model_params["name"] == "random":
        model = load_sirl(ckpt_path)
    if model_params["name"] == "pca":
        # assumes ckpt_path points to .joblib file,
        model = load_pca(ckpt_path)
        return model, ckpt_path
    if model_params["name"] == "sirl":
        model = load_sirl(ckpt_path)
    if model_params["name"] == "tversky_sirl":
        model = load_tversky_sirl(ckpt_path)
    if model_params["name"] == "tversky_sirl_2":
        model = load_tversky_sirl_2(ckpt_path)
    if model_params["name"] == "tversky_proj":
        model = load_tversky_proj(ckpt_path)
    if model_params["name"] == "tversky_vae":
        model = load_tversky_vae(ckpt_path)
    return model


def train_model(config, data, results_dir, seed):
    """
    extract anchors, positives, negatives for training
    train model
    save model checkpoint, return model and checkpoint path
    """
    anchors = data["anchors"]
    positives = data["positives"]
    negatives = data["negatives"]

    model_params = config["model"]
    # random
    if model_params["name"] == "random":
        model = init_random_sirl(config, anchors, positives, negatives)
        ckpt_path = str(results_dir / f"random_dim{model.encoder[-1].out_features}_seed{seed}.pth")
        model.save_model(ckpt_path)
    # PCA
    if model_params["name"] == "pca":
        model = fit_pca(config, anchors, positives, negatives)
        ckpt_path = save_pca(model, results_dir)
        return model, ckpt_path
    # SIRL
    if model_params["name"] == "sirl":
        model, history = train_sirl(config, anchors, positives, negatives)
        ckpt_path = str(results_dir / f"sirl_dim{model.encoder[-1].out_features}_seed{seed}.pth")
        model.save_model(ckpt_path)
    # Tversky SIRL (TverskySimilarity in triplet loss)
    if model_params["name"] == "tversky_sirl":
        model, history = train_tversky_sirl(config, anchors, positives, negatives)
        ckpt_path = str(results_dir / f"tversky_sirl_dim{model.encoder[-1].out_features}_fbank{model_params["fbank_size"]}_seed{seed}.pth")
        model.save_model(ckpt_path)
    # Tversky SIRL 2 (TverskyProjection instead of MLP, TverskySimilarity in triplet loss)
    if model_params["name"] == "tversky_sirl_2":
        model, history = train_tversky_sirl_2(config, anchors, positives, negatives)
        ckpt_path = str(results_dir / f"tversky_sirl_2_dim{model_params["latent_dim"]}_fbank{model_params["fbank_size"]}_seed{seed}.pth")
        model.save_model(ckpt_path)
    # tversky projection layer encoder with normal triplet loss
    if model_params["name"] == "tversky_proj":
        model, history = train_tversky_proj(config, anchors, positives, negatives)
        unix_timestamp = int(time.time())
        ckpt_path = str(results_dir / f"tversky_proj_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
    if model_params["name"] == "tversky_vae":
        # VAE is unsupervised: train on the raw (unlabelled) trajectories
        model, history = train_tversky_vae(config, data["trajs"])
        unix_timestamp = int(time.time())
        ckpt_path = str(results_dir / f"tversky_vae_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
    return model, ckpt_path

