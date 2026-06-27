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
    pass


def train_model(config, data):
    """
    extract anchors, positives, negatives for training
    """
    anchors = data["anchors"]
    positives = data["positives"]
    negatives = data["negatives"]

    model_params = config["model"]
    if model_params["name"] == "pca":
        model = fit_pca(config, anchors, positives, negatives)
    
    # TODO save model checkpoint
    return model


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