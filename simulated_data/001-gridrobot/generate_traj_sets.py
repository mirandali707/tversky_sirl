"""
Generate synthetic gridrobot trajectories together with simulated "human"
labels (rewards, pairwise preferences, and similarity triplets).

Each trajectory is 19-dimensional: 18 spatial coordinates (9 waypoints) plus a
trailing end-effector joint angle. Everything needed lives in this directory --
no `src.*` imports.

Example:
    python generate_traj_sets.py --samples 1960 --num-prefs 500 --visualize
"""
import argparse
import json
import os
import random

import numpy as np

from gridrobot import Gridrobot
from human import GridrobotHuman


# Default 5x5 obstacle-free world; the four corners are starts heading to the
# opposite corner. Mirrors config/envs/gridrobot.yaml and the first human in
# config/humans/gridrobot.yaml.
DEFAULT_CONFIG = {
    "X": 5,
    "Y": 5,
    "obstacles": [],
    "starts": [[0, 0], [0, 4], [4, 0], [4, 4]],
    "goals": [[4, 4], [4, 0], [0, 4], [0, 0]],
    "features": ["computer_dist", "joint_up"],
    # One reward weight per feature.
    "theta": [-10.0, -10.0],
    "beta": 10.0,
    "feature_scaling": "normalize",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                        help="optional JSON file overriding DEFAULT_CONFIG")
    parser.add_argument("--samples", type=int, default=100,
                        help="number of trajectories to sample")
    parser.add_argument("--num-prefs", type=int, default=500,
                        help="number of preference pairs to label (0 to skip)")
    parser.add_argument("--num-triplets", type=int, default=0,
                        help="number of similarity triplets to label (0 to skip)")
    parser.add_argument("--noiseless", action="store_true",
                        help="label deterministically instead of Boltzmann-rationally")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="data")
    parser.add_argument("--visualize", action="store_true",
                        help="save a plot of the highest-reward trajectories")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config is not None:
        with open(args.config) as f:
            config.update(json.load(f))

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 1. Build the environment and sample trajectories.
    env = Gridrobot(config["X"], config["Y"], config["obstacles"],
                    config["starts"], config["goals"])
    trajs = np.array(env.generate_trajs(args.samples))
    print("generated {} trajectories of dim {} ({} waypoints + joint angle)".format(
        len(trajs), trajs.shape[1], (trajs.shape[1] - 1) // 2))

    # 2. Build the synthetic human and featurize / score the trajectories.
    human = GridrobotHuman(env, features=config["features"], theta=config["theta"],
                           beta=config["beta"], feature_scaling=config["feature_scaling"],
                           rng=rng)
    human.set_trajset(trajs)
    rewards = human.reward_labels()

    # 3. Generate human preference / triplet labels.
    bundle = {
        "trajs": trajs,
        "features": human.featurized_trajs,
        "rewards": rewards,
        "theta": np.asarray(config["theta"], dtype=float),
        "beta": np.asarray(config["beta"], dtype=float),
    }
    if args.num_prefs > 0:
        pairs, labels = human.generate_preference_labels(
            args.num_prefs, noisy=not args.noiseless)
        bundle["pref_pairs"] = pairs
        bundle["pref_labels"] = labels
        print("labeled {} preference pairs".format(len(labels)))
    if args.num_triplets > 0:
        triplets, tlabels = human.generate_triplet_labels(args.num_triplets)
        bundle["triplets"] = triplets
        bundle["triplet_labels"] = tlabels
        print("labeled {} similarity triplets".format(len(tlabels)))

    # 4. Save everything to a single .npz bundle.
    here = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(here, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "gridrobot_{}.npz".format(len(trajs)))
    np.savez(out_path, **bundle)
    print("saved labeled dataset -> {}".format(out_path))

    if args.visualize:
        im_path = os.path.join(save_dir, "gridrobot_{}.png".format(len(trajs)))
        env.visualize(trajs, rewards, im_path=im_path)
        print("saved visualization -> {}".format(im_path))


if __name__ == "__main__":
    main()
