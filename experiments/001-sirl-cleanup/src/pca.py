from utils import config_overridable
import joblib
import numpy as np
from sklearn.decomposition import PCA

@config_overridable
def fit_pca(config, anchors, positives, negatives, n_components=6, random_state=42):
    """
    combine anchors, positives, negatives 
    (for comparison to SIRL-type methods which actually use the triplet info)
    and fit PCA
    """
    query_trajs = np.concatenate([anchors, positives, negatives], axis=0)
    X = query_trajs.reshape(len(query_trajs), -1)  # (N, 21*97) = (N, 2037)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    print("pca fit")
    return pca


def save_pca(model, results_dir):
    ckpt_path = str(results_dir / f"pca_dim{model.n_components_}.joblib")
    joblib.dump(model, ckpt_path)
    print(f"saved pca to {ckpt_path}")
    return ckpt_path


def load_pca(ckpt_path):
    return joblib.load(ckpt_path)