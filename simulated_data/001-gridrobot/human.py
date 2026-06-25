import numpy as np


class GridrobotHuman:
    """
    A simulated ("synthetic") human that labels gridrobot trajectories.

    Trajectories are 19-dimensional: 18 spatial coordinates (9 waypoints) plus a
    trailing joint angle. The human scores a trajectory with a linear reward over
    two hand-crafted features:

        reward(traj) = - theta . phi(traj)

      - computer_dist: summed distance from each waypoint to the grid center
                       (the "computer" the robot reaches toward).
      - joint_up:      magnitude of the end-effector joint angle.

    and answers queries Boltzmann-rationally with inverse-temperature `beta`
    (higher beta = more deterministic / less noisy).

    Label types produced:
      - reward labels:      a scalar reward per trajectory.
      - preference labels:  for a pair (i, j), 0 if i is preferred else 1.
      - triplet labels:     for (anchor, i, j), which of i / j is more similar
                            to the anchor in feature space.
    """

    def __init__(self, env, features=("computer_dist", "joint_up"),
                 theta=None, beta=10.0, feature_scaling="normalize", rng=None):
        self.env = env
        self.features = list(features)
        self.feat_scaling = feature_scaling
        self.beta = beta
        self.rng = rng if rng is not None else np.random.default_rng()

        # Each feature contributes one scalar, so theta has one weight per feature.
        n_feats = len(self.features)
        if theta is None:
            theta = np.ones(n_feats)
        self.theta = np.asarray(theta, dtype=float)
        assert len(self.theta) == n_feats, \
            "theta has {} entries, expected {}".format(len(self.theta), n_feats)

        self.scaling_coeffs = []
        self.all_trajs = None
        self.featurized_trajs = None

    # ------------------------------------------------------------------ #
    # Featurization
    # ------------------------------------------------------------------ #
    def set_trajset(self, all_trajs):
        """Register the trajectory set and fit feature-scaling statistics."""
        self.all_trajs = np.asarray(all_trajs)
        self.scaling_coeffs = []
        raw = np.array([self._raw_features(traj) for traj in self.all_trajs])
        for col in raw.T:
            if self.feat_scaling == "standardize":
                self.scaling_coeffs.append({"mu": col.mean(), "sigma": col.std() + 1e-8})
            elif self.feat_scaling == "normalize":
                self.scaling_coeffs.append({"min": col.min(), "max": col.max()})
            else:
                self.scaling_coeffs.append({})
        self.featurized_trajs = np.array([self.calc_features(traj) for traj in self.all_trajs])
        return self.featurized_trajs

    def calc_features(self, traj):
        feats = self._raw_features(traj)
        if not self.scaling_coeffs:
            return feats
        scaled = np.empty_like(feats)
        for i, (f, c) in enumerate(zip(feats, self.scaling_coeffs)):
            if self.feat_scaling == "normalize":
                denom = (c["max"] - c["min"]) or 1.0
                scaled[i] = (f - c["min"]) / denom
            elif self.feat_scaling == "standardize":
                scaled[i] = (f - c["mu"]) / c["sigma"]
            else:
                scaled[i] = f
        return scaled

    def _raw_features(self, traj):
        feats = []
        for feat in self.features:
            if feat == "computer_dist":
                feats.append(self._computer_dist_feature(traj))
            elif feat == "joint_up":
                feats.append(self._joint_feature(traj))
            else:
                raise NotImplementedError("unknown feature '{}'".format(feat))
        return np.asarray(feats, dtype=float)

    def _computer_dist_feature(self, traj):
        """Summed distance from every waypoint to the grid center."""
        cx = (self.env.X - 1) / 2
        cy = (self.env.Y - 1) / 2
        total = 0.0
        # Stride over (x, y) pairs only; the trailing element is the joint angle.
        for i in range(0, len(traj) - 1, 2):
            total += np.hypot(traj[i] - cx, traj[i + 1] - cy)
        return total

    @staticmethod
    def _joint_feature(traj):
        """Magnitude of the (constant) end-effector joint angle."""
        return abs(traj[-1])

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #
    def reward(self, traj):
        """Ground-truth (noise-free) reward for a trajectory."""
        return -float(np.dot(self.calc_features(traj), self.theta))

    def reward_labels(self, trajs=None):
        trajs = self.all_trajs if trajs is None else trajs
        return np.array([self.reward(traj) for traj in trajs])

    def query_pref(self, traj1, traj2, noisy=True):
        """
        Preference label for a pair. Returns 0 if traj1 is preferred, else 1.
        With `noisy=True` the answer is sampled Boltzmann-rationally; otherwise
        the higher-reward trajectory is returned deterministically.
        """
        r1, r2 = self.reward(traj1), self.reward(traj2)
        if not noisy:
            if r1 == r2:
                return self.rng.integers(2)
            return 0 if r1 > r2 else 1
        # P(choose i) propto exp(beta * reward_i).
        logits = self.beta * np.array([r1, r2])
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return int(self.rng.choice([0, 1], p=p))

    def generate_preference_labels(self, num_pairs, noisy=True):
        """
        Sample `num_pairs` random trajectory pairs and label each one.
        Returns (pairs, labels) where pairs[k] = (i, j) are indices into
        `self.all_trajs` and labels[k] in {0, 1}.
        """
        assert self.all_trajs is not None, "call set_trajset() first."
        n = len(self.all_trajs)
        pairs = np.empty((num_pairs, 2), dtype=int)
        labels = np.empty(num_pairs, dtype=int)
        for k in range(num_pairs):
            i, j = self.rng.choice(n, size=2, replace=False)
            pairs[k] = (i, j)
            labels[k] = self.query_pref(self.all_trajs[i], self.all_trajs[j], noisy=noisy)
        return pairs, labels

    def query_triplet(self, anchor, traj1, traj2):
        """Return 0 if traj1 is more similar to `anchor` than traj2, else 1."""
        fa = self.calc_features(anchor)
        f1 = self.calc_features(traj1)
        f2 = self.calc_features(traj2)
        d1 = np.sum((fa - f1) ** 2)
        d2 = np.sum((fa - f2) ** 2)
        return 0 if d1 <= d2 else 1

    def generate_triplet_labels(self, num_triplets):
        """
        Sample `num_triplets` (anchor, i, j) index triplets and label each by
        which of i / j is more similar to the anchor in feature space.
        Returns (triplets, labels), labels[k] in {0, 1}.
        """
        assert self.all_trajs is not None, "call set_trajset() first."
        n = len(self.all_trajs)
        triplets = np.empty((num_triplets, 3), dtype=int)
        labels = np.empty(num_triplets, dtype=int)
        for k in range(num_triplets):
            a, i, j = self.rng.choice(n, size=3, replace=False)
            triplets[k] = (a, i, j)
            labels[k] = self.query_triplet(
                self.all_trajs[a], self.all_trajs[i], self.all_trajs[j])
        return triplets, labels
