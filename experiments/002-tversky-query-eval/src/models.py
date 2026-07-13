# from pca import *
# from sirl import train_sirl, load_sirl, init_random_sirl
# from tversky_sirl import train_tversky_sirl, load_tversky_sirl
# from tversky_sirl_2 import train_tversky_sirl_2, load_tversky_sirl_2
from tversky_proj import train_tversky_proj, load_tversky_proj
import time


def load_model(train_config, ckpt_path):
    """
    loads a saved model checkpoint.
    """
    model_params = train_config["model"]
    if model_params["name"] == "tversky_proj":
        model = load_tversky_proj(ckpt_path)
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
    # tversky projection layer encoder with normal triplet loss
    if model_params["name"] == "tversky_proj":
        model, history = train_tversky_proj(config, anchors, positives, negatives)
        unix_timestamp = int(time.time())
        ckpt_path = str(results_dir / f"tversky_proj_{unix_timestamp}.pth")
        model.save_model(ckpt_path)
    return model, ckpt_path

