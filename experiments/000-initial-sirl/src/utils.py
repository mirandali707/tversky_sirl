import random
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.abspath('../..'))
from src.utils.input_utils import transform_input
from src.envs.jacorobot import Jacorobot
from src.utils.bullet_utils import waypts_to_xyz
from src.envs.jacorobot import Jacorobot
from src.models.humans.jacorobot_human import JacorobotHuman

from fpe import FPEProbe

DATA_DIR = "../../data/Data Uploads"

INPUT_DICT = {'lowdim': False, 'EErot': False, 'noxyz': False, 'norot': True, 'noangles': True}
def tx(trajs):
    """Apply the JacoRobot input transform to a (N, 21, 97) trajectory array."""
    x = torch.as_tensor(trajs, dtype=torch.float32)
    return transform_input(x, INPUT_DICT).numpy()


def load_data(n_queries, data_dir=DATA_DIR, per_file=100, start_from_idx = 0):
    """
    loads query data, pass in number of queries from 100-1000
    """
    assert n_queries % per_file == 0, f"n_queries must be a multiple of {per_file}"

    n_files = n_queries // per_file

    all_anchors, all_pos, all_neg = [], [], []
    for i in range(n_files):
        path = f"{data_dir}/sim_{per_file}_{i + start_from_idx}.npz"
        with np.load(path) as data:
            all_anchors.append(data['anchors'])
            all_pos.append(data['positives'])
            all_neg.append(data['negatives'])

    # concat queries from all files
    anchors, positives, negatives = (np.concatenate(all_anchors, axis=0), np.concatenate(all_pos, axis=0), np.concatenate(all_neg, axis=0))
    # jacorobot transform
    # NOTE could add flag for whether to norm or not
    anchors, positives, negatives = tx(anchors), tx(positives), tx(negatives)
    return anchors, positives, negatives

def make_env(params):
    return Jacorobot(params["object_centers"], params["resources_dir"], params["horizon"], params["timestep"], params["real"])


def make_human(params, env, trajs):
    human = JacorobotHuman(params, env)
    human.set_trajset(trajs)
    if "preferencer" in params:
        human.set_preference(params["preferencer"])
    return human

def load_all_trajs():
    all_trajs = np.load(F"{DATA_DIR}/jacorobot_10000.npy")
    print(f"all_trajs shape: {all_trajs.shape}")
    return all_trajs

def get_features(all_trajs, traj_list):
    """
    all_trajs is the list of all jacorobot trajectories, for normalization. shape (1000, 21, 97)
    traj_list is the list of trajectories we want to get the features for. shape (n_trajs, 21, 97)
    returns np array of shape (n_trajs, 2) since we are currently looking at "table" and "laptop" features only
    """
    env_params = {
        "type": "jacorobot",
        "resources_dir": "../data/resources",
        "object_centers": {'HUMAN_CENTER': [-0.2, -0.6, 0.9], 'LAPTOP_CENTER': [-0.5, 0.0, 0.635]},
        "horizon": 10,
        "timestep": 0.5,
        "per_SG": 100,
        "trajset_file": "../../data/traj_sets/jacorobot_10000.npy",
        "train_test_split": 0.8,
        "real": False,
        "input_dict": {'lowdim': False, 'EErot': False, 'noxyz': False, 'norot': True, 'noangles': True}
    }
    human_params = {
        "type": "jacorobot",
        # "features": ["table", "coffee", "laptop", "proxemics"],
        "features": ["table", "laptop"],
        "feature_scaling": "normalize",
    }

    env = make_env(env_params)
    human = make_human(human_params, env, all_trajs)

    return np.array([human.calc_features(traj) for traj in traj_list])

# code stolen / shortened from:
# src/eval/train_preference_jacorobot.ipynb
# parser.py
def get_train_test_trajs_feats(train_test_split=0.8, seed=0):
    all_trajs = load_all_trajs()
    all_features = get_features(all_trajs, all_trajs)

    # Train/test split with a fixed seed so the split is identical across runs
    n = len(all_trajs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_test_split)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_trajs = all_trajs[train_idx]
    train_features = all_features[train_idx]
    test_trajs = all_trajs[test_idx]
    test_features = all_features[test_idx]

    print(f"train: {train_trajs.shape}, {train_features.shape}")
    print(f"test:  {test_trajs.shape}, {test_features.shape}")

    # apply Jacorobot transforms
    train_trajs = tx(train_trajs)
    test_trajs = tx(test_trajs)

    return train_trajs, train_features, test_trajs, test_features


def eval_fpe(model, train_trajs, train_features, test_trajs, test_features):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        Z_train = model(torch.as_tensor(train_trajs, dtype=torch.float32, device=device)).cpu().numpy()
        Z_test  = model(torch.as_tensor(test_trajs,  dtype=torch.float32, device=device)).cpu().numpy()

    reg = LinearRegression().fit(Z_train, train_features)
    pred = reg.predict(Z_test)
    fpe = np.mean(np.sum((pred - test_features) ** 2, axis=1))
    return fpe

import torch.nn as nn

class StandardizedFeaturizer(nn.Module):
    """Wraps a featurizer so its embeddings are z-scored with fixed train stats."""
    def __init__(self, base, mu, sd):
        super().__init__()
        self.base = base
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
    def forward(self, x):
        return (self.base(x) - self.mu) / self.sd


def eval_fpe_probe(model, train_trajs, train_features, test_trajs, test_features):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    latent_dim = model(torch.as_tensor(train_trajs[:1], dtype=torch.float32, device=device)).shape[-1]
    num_features = train_features.shape[-1]

    # train-set stats ONLY (never fit on test)
    with torch.no_grad():
        Z_train = model(torch.as_tensor(train_trajs, dtype=torch.float32, device=device))
    mu = Z_train.mean(0)
    sd = Z_train.std(0) + 1e-8

    std_model = StandardizedFeaturizer(model, mu, sd).to(device).eval()

    probe = FPEProbe(std_model, latent_dim=latent_dim, num_features=num_features)
    probe.train(train_trajs, train_features, test_trajs=test_trajs, test_features=test_features)
    # return probe.evaluate(test_trajs, test_features)
    return probe.evaluate(test_trajs, test_features), probe


def fit_pca(query_trajs, n_components=6, random_state=42):
    """Fit PCA on flattened trajectories. Call once."""
    X = query_trajs.reshape(len(query_trajs), -1)  # (N, 21*97) = (N, 2037)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    return pca


def eval_pca_fpe(pca, train_trajs, train_features, test_trajs, test_features):
    """Compute FPE of a PCA representation via linear probe."""
    Z_train = pca.transform(train_trajs.reshape(len(train_trajs), -1))
    Z_test  = pca.transform(test_trajs.reshape(len(test_trajs), -1))

    reg = LinearRegression().fit(Z_train, train_features)
    pred = reg.predict(Z_test)
    fpe = np.mean(np.sum((pred - test_features) ** 2, axis=1))
    return fpe


def set_all_seeds(seed):
    """Set seeds for every source of randomness in SIRL training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vis_trajectory(traj, title=""):
    env_params = {
        "object_centers": {'HUMAN_CENTER': [-0.2, -0.6, 0.9], 'LAPTOP_CENTER': [-0.5, 0.0, 0.635]},
        "resources_dir": "../data/resources",
        "horizon": 10,
        "timestep": 0.5,
        "real": False,
    }

    env = Jacorobot(
        env_params["object_centers"],
        env_params["resources_dir"],
        env_params["horizon"],
        env_params["timestep"],
        debug=False  # headless — no GUI needed
    )

    xyz = waypts_to_xyz(env.objectID["robot"], traj)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='lines+markers'
    ))

    posH = env_params["object_centers"]["HUMAN_CENTER"]
    posL = env_params["object_centers"]["LAPTOP_CENTER"]

    fig.add_trace(go.Scatter3d(
        x=[posH[0], posL[0]],
        y=[posH[1], posL[1]],
        z=[posH[2], posL[2]],
        mode='markers',
        marker=dict(
            color=["gray", "black"],
            symbol=["cross", "square"],
            size=8,
            showscale=False
        ),
        name="human (+), laptop (square)"
    ))
    fig.update_layout(title=title)

    fig.show()
