import numpy as np
import time

from learner import Learner
from multiprocessing import Pool


def parallelized_preferencer(preferencer, Phi_R, Phi_S, theta, beta):
    return preferencer.observation_model([Phi_R], [Phi_S], theta, beta)


class BaseHuman:
    def __init__(self):
        self.num_triplet_queries = 0
        self.num_pref_queries = 0
        self.triplet_times = []
        self.pref_times = []
        self.num_cores = 4
        self.record = False

    def set_trajset(self, all_trajs):
        self.scaling_coeffs = []
        self.all_trajs = all_trajs
        self.feat_scale_construct(all_trajs)

        self.featurized_trajs = np.array([self.calc_features(traj) for traj in all_trajs])

    def set_preference(self, params):
        assert hasattr(self, 'features'), "Human does not have features yet."
        self.theta = params["theta"]
        self.beta = params["beta"]

        self.preferencer = Learner(self.featurized_trajs, self.featurized_trajs, [self.theta], [self.beta], params["f_method"], params["s_method"])
        #with Pool(processes=self.num_cores) as pool:
        #    self.probs = pool.starmap(parallelized_preferencer,
        #                              [(self.preferencer, R, S, self.theta, self.beta) for R, S in zip(self.preferencer.Phi_xibar_R, self.preferencer.Phi_xibar_S)])

    def generate_demos(self, samples):
        """
        Output one or more simulated human train_trajs.
        Params:
            samples [int] -- number of train_trajs to sample.
        Return:
            trajs [list] -- Sampled train_trajs. A trajectory is a list of state numbers.
        """
        assert hasattr(self, 'preferencer'), "Human does not have a preferencer yet."
        traj_idx = np.random.choice(len(self.probs), samples, p=self.probs)
        return np.array([self.all_trajs[i] for i in traj_idx])

    def query_pref(self, sample1, sample2, validate=False):
        assert hasattr(self, 'preferencer'), "Human does not have a preferencer yet."
        if not validate:
            self.num_pref_queries += 1

        sample1_features = np.array(self.calc_features(sample1))
        sample2_features = np.array(self.calc_features(sample2))

        sample1_reward = np.exp(-self.beta * np.dot(sample1_features, self.theta))
        sample2_reward = np.exp(-self.beta * np.dot(sample2_features, self.theta))
        if validate:
            if sample1_reward == sample2_reward:
                return 0.5
            return 0 if sample1_reward > sample2_reward else 1

        metadata = {"gt": "{:.3f}, {:.3f}".format(np.dot(sample1_features, self.theta), np.dot(sample2_features, self.theta)) if not self.record else None,
                    "prefq": self.num_pref_queries}
        if self.env.debug and not validate:
            start = time.time()
            ans = self.env.query_pref(sample1, sample2, metadata=metadata)
            self.pref_times.append(time.time() - start)
            return ans
        return np.random.choice([0, 1], 1, p=[sample1_reward / (sample1_reward + sample2_reward), sample2_reward / (sample1_reward + sample2_reward)])[0]

    def query_triplet(self, anchor, sample1, sample2, validate=False):
        assert hasattr(self, 'features'), "Human does not have features yet."
        if not validate:
            self.num_triplet_queries += 1
        anchor_features = np.array(self.calc_features(anchor))
        sample1_features = np.array(self.calc_features(sample1))
        sample2_features = np.array(self.calc_features(sample2))

        sim1 = -np.sum((anchor_features - sample1_features) ** 2)
        sim2 = -np.sum((anchor_features - sample2_features) ** 2)
        sim3 = -np.sum((sample1_features - sample2_features) ** 2)
        if sim1 > sim2:
            gt = 1
        else:
            gt = 2

        metadata = {"gt": "{:.3f}, {:.3f}, {:.3f}".format(-sim1, -sim2, -sim3) if not self.record else None,
                    "simq": self.num_triplet_queries}
        if self.env.debug and not validate:
            start = time.time()
            ans = self.env.query_triplet(anchor, sample1, sample2, metadata=metadata)
            self.triplet_times.append(time.time() - start)
            return ans
        return (anchor, sample1, sample2) if gt == 1 else (anchor, sample2, sample1)

    def feat_scale_construct(self, all_trajs):
        assert hasattr(self, 'features'), "Human does not have features yet."
        featurized_trajs = np.array([self.calc_features(traj) for traj in all_trajs])
        for Phi in featurized_trajs.T:
            coeffs = {}
            if self.feat_scaling == "standardize":
                mu = np.mean(Phi)
                s = np.std(Phi)
                coeffs = {"mu": mu, "sigma": s}
            elif self.feat_scaling == "normalize":
                min_val = min(Phi)
                max_val = max(Phi)
                coeffs = {"min": min_val, "max": max_val}
            self.scaling_coeffs.append(coeffs)

    def calc_features(self, traj):
        raise NotImplementedError

    def calc_rewards(self, trajs):
        # Return rewards without beta term
        assert hasattr(self, 'preferencer'), "Human does not have a preferencer yet."
        featurized_trajs = np.array([self.calc_features(traj) for traj in trajs])
        return self.preferencer.reward_model(featurized_trajs, featurized_trajs, self.theta, 1)

    def calc_probs(self, trajs):
        assert hasattr(self, 'preferencer'), "Human does not have a preferencer yet."
        featurized_trajs = np.array([self.calc_features(traj) for traj in trajs])
        probs = [self.preferencer.observation_model([Phi_R], [Phi_S], self.theta, self.beta) for Phi_R, Phi_S in zip(featurized_trajs, featurized_trajs)]
        return np.array(probs)

    def reset(self):
        self.num_triplet_queries = 0
        self.num_pref_queries = 0

    def __str__(self):
        raise NotImplementedError
