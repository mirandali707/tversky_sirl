import numpy as np
from sklearn.decomposition import PCA


def get_model(config, data):
    model_params = config["model"]
    if model_params["load_ckpt"]:
        return load_model(config)
    # if not loading model from ckpt, train model from scratch
    return train_model(config, data)


def load_model(config):
    # TODO load model
    model_params = config["model"]
    ckpt_path = model_params["ckpt_path"]
    if model_params["name"] == "pca":
        model = None # TODO load pca sklearn style
        return model, ckpt_path
    model = None # TODO load model
    return model, ckpt_path


def train_model(config, data):
    """
    extract anchors, positives, negatives for training
    train model
    save model checkpoint, return model and checkpoint path
    """
    anchors = data["anchors"]
    positives = data["positives"]
    negatives = data["negatives"]

    model_params = config["model"]
    if model_params["name"] == "pca":
        model = fit_pca(config, anchors, positives, negatives)
        # TODO save pca to {results_dir}/pca_{other params????? embed dim?}
        ckpt_path = "PCA_PATH_TEMP"
        return model, ckpt_path

    # TODO save model checkpoint
    # model.save_model(f"{OUT_DIR}/tversky_4.pth")
    ckpt_path = "TEMP"
    return model, ckpt_path


def fit_pca(config, anchors, positives, negatives, n_components=6, random_state=42):
    """
    combine anchors, positives, negatives 
    (for comparison to SIRL-type methods which actually use the triplet info)
    and fit PCA
    """
    if "n_components" in config: 
        n_components = config["n_components"]
    if "random_state" in config: 
        n_components = config["random_state"]

    query_trajs = np.concatenate([anchors, positives, negatives], axis=0)
    X = query_trajs.reshape(len(query_trajs), -1)  # (N, 21*97) = (N, 2037)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    print("pca fit")
    return pca