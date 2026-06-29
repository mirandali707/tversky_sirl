from pca import *
from sirl import train_sirl, load_sirl, init_random_sirl
from tversky_sirl import train_tversky_sirl, load_tversky_sirl
from tversky_sirl_2 import train_tversky_sirl_2, load_tversky_sirl_2


def get_model(config, data, results_dir, seed):
    """
    loads model from checkpoint if "load_ckpt" is True, 
    otherwise trains the model and saves it.
    returns (model, ckpt_path) in both cases
    """
    model_params = config["model"]
    if model_params["load_ckpt"]:
        return load_model(config)
    # if not loading model from ckpt, train model from scratch
    return train_model(config, data, results_dir, seed)


def load_model(config):
    """
    loads a saved model checkpoint.
    - if model is PCA (config["model"]["name"] == "pca"), loads from .joblib
    """
    model_params = config["model"]
    ckpt_path = model_params["ckpt_path"]
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
    return model, ckpt_path


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
    # random baseline (untrained sirl)
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
        # TODO save history? really i should learn how to use wandb
    # Tversky SIRL (TverskySimilarity in triplet loss)
    if model_params["name"] == "tversky_sirl":
        model, history = train_tversky_sirl(config, anchors, positives, negatives)
        ckpt_path = str(results_dir / f"tversky_sirl_dim{model.encoder[-1].out_features}_fbank{model_params["fbank_size"]}_seed{seed}.pth")
        model.save_model(ckpt_path)
        # TODO save history? really i should learn how to use wandb
    # Tversky SIRL 2 (TverskyProjection instead of MLP, TverskySimilarity in triplet loss)
    if model_params["name"] == "tversky_sirl_2":
        model, history = train_tversky_sirl_2(config, anchors, positives, negatives)
        ckpt_path = str(results_dir / f"tversky_sirl_2_dim{model_params["latent_dim"]}_fbank{model_params["fbank_size"]}_seed{seed}.pth")
        model.save_model(ckpt_path)
        # TODO save history? really i should learn how to use wandb
    return model, ckpt_path

