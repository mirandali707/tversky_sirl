import numpy as np
from sklearn.decomposition import PCA

def train_model(anchors, positives, negatives, config):
    model_params = config["model"]
    if model_params["name"] == "pca":
        model = fit_pca(anchors, positives, negatives, config)


def fit_pca(anchors, positives, negatives, config, n_components=6, random_state=42):
    """
    combine anchors, positives, negatives and fit PCA
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