from models import load_model_only
from utils import *

# TODO eventually write a script to loop all *tversky* ckpts
config = parse_config('../configs/tversky_sirl.yaml')
model = load_model_only(config, "../results/tversky_sirl_gridrobot/tversky_sirl_dim10_fbank4_seed0.pth") # wtf lol why am i returning ckpt path

feature_bank = model.tversky_sim.feature_bank.weight.detach() # 4 Tversky features, of 10-dim embeddings

data = load_data(config)
all_trajs = data["trajs"] # (1960, 19)
all_feats = data["features"] # (1960, 2)
# from tversky/test_mnist.py
# edited
import torch.nn.functional as F
def compute_salience(x, feature_bank):
    """
    x is (N, D)
    feature_bank is (F, D)
    """
    feature_measures = x @ feature_bank.T # (N, F)
    salience_measures = F.relu(feature_measures).sum(-1)
    return salience_measures

all_embeds = model(torch.as_tensor(all_trajs, dtype=torch.float32)) # (1960, 10)
all_salience = compute_salience(all_embeds, feature_bank).detach() # (1960)
mu = all_embeds.mean(0)        # shape [6]
centered_embeds = all_embeds - mu     # [10000, 6]
centered_salience = compute_salience(centered_embeds, feature_bank).detach()

min_salience_idx = np.argmin(centered_salience).item()
min_salience_traj = all_trajs[min_salience_idx]
max_salience_idx = np.argmax(centered_salience).item()
max_salience_traj = all_trajs[max_salience_idx]
print(f"min salience traj has features {all_feats[min_salience_idx]}")
print(f"max salience traj has features {all_feats[max_salience_idx]}")

import sys
sys.path.insert(0, '../../../simulated_data/001-gridrobot/')
from gridrobot import Gridrobot
    
config = {
    "X": 5,
    "Y": 5,
    "obstacles": [],
    "starts": [[0, 0], [0, 4], [4, 0], [4, 4]],
    "goals": [[4, 4], [4, 0], [0, 4], [0, 0]],
    "features": ["computer_dist", "joint_up"],
    "thetas": [[ -10.0, -10.0 ],
              [ 0.0, -10.0 ],
              [ 10.0, -10.0 ],
              [ -10.0, 0.0 ],
              [ 10.0, 0.0 ],
              [ -10.0, 10.0 ],
              [ 0.0, 10.0 ],
              [ 10.0, 10.0 ]
    ],
    "beta": 10.0,
    "feature_scaling": "normalize",
    "train_test_split": 0.8
}
env = Gridrobot(config["X"], config["Y"], config["obstacles"], config["starts"], config["goals"])

env.visualize_one_traj(min_salience_traj)
env.visualize_one_traj(max_salience_traj)