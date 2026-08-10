from base_human import BaseHuman
from bullet_utils import *


class JacorobotHuman(BaseHuman):
    def __init__(self, params, env):
        super().__init__()
        # Set environment and features in the environment.
        self.env = env
        self.features = params["features"]
        self.feat_scaling = params["feature_scaling"]
        self.theta = None

    def __str__(self):
        return "JacorobotHuman " + str(self.theta)

    def calc_features(self, traj):
        assert hasattr(self, 'features'), "Human does not have features yet."
        features = self.calc_features_cached(traj)
        if len(self.scaling_coeffs) != 0:
            for feat in range(len(features)):
                if self.feat_scaling == "normalize":
                    features[feat] = (features[feat] - self.scaling_coeffs[feat]["min"]) / (self.scaling_coeffs[feat]["max"] - self.scaling_coeffs[feat]["min"])
                elif self.feat_scaling == "standardize":
                    features[feat] = (features[feat] - self.scaling_coeffs[feat]["mu"]) / self.scaling_coeffs[feat]["sigma"]
        return np.array(features)

    def calc_features_cached(self, traj):
        traj = np.array(traj)
        features = []
        for feat in self.features:
            if feat == "table":
                features.extend(self.table_features(traj))
            elif feat == "human":
                features.extend(self.human_features(traj))
            elif feat == "laptop":
                features.extend(self.laptop_features(traj))
            elif feat == "proxemics":
                features.extend(self.proxemics_features(traj))
            elif feat == "coffee":
                features.extend(self.coffee_features(traj))
            else:
                raise NotImplementedError
        return features

    # -- Distance to Table -- #

    def table_features(self, traj):
        """
        Computes the total feature value over waypoints based on z-axis distance to table.
        ---
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar feature sum
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            # joints = np.hstack((waypt[:7], np.array([0, 0, 0])))
            # move_robot(self.env.objectID["robot"], joint_poses=joints)
            # coords = robot_coords(self.env.objectID["robot"])
            # posT, _ = p.getBasePositionAndOrientation(self.env.objectID["stand"])
            # feat_val += coords[6][2] - posT[2]
            # print(self.stand_z, posT[2])
            stand_z = p.getBasePositionAndOrientation(self.env.objectID["stand"])[0][2]
            feat_val += waypt[90] - stand_z
        return feat_val

    # -- Distance to Laptop -- #

    def laptop_features(self, traj):
        """
        Computes distance from end-effector to laptop in xy coords.
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar distance sum where
                0: EE is at more than 0.3 meters away from laptop
                +: EE is closer than 0.3 meters to laptop
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            # joints = np.hstack((waypt[:7], np.array([0, 0, 0])))
            # move_robot(self.env.objectID["robot"], joint_poses=joints)
            # coords = robot_coords(self.env.objectID["robot"])
            # EE_coord_xy = coords[6][0:2]
            # posL, _ = p.getBasePositionAndOrientation(self.env.objectID["laptop"])
            # laptop_xy = posL[0:2]
            # dist = np.linalg.norm(EE_coord_xy - laptop_xy) - 0.3
            # feat_val += -((dist < 0) * dist)
            dist = np.linalg.norm(waypt[88:90] - waypt[94:96]) - 0.8
            feat_val += -((dist < 0) * dist)
        return feat_val

    # -- Distance to Human -- #

    def human_features(self, traj):
        """
        Computes distance from end-effector to human in xy coords.
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar distance sum where
                0: EE is at more than 0.3 meters away from human
                +: EE is closer than 0.3 meters to human
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            waypt = np.hstack((waypt[:7], np.array([0, 0, 0])))
            move_robot(self.env.objectID["robot"], joint_poses=waypt)
            coords = robot_coords(self.env.objectID["robot"])
            EE_coord_xy = coords[6][0:2]
            posH, _ = p.getBasePositionAndOrientation(self.env.objectID["human"])
            human_xy = posH[0:2]
            dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.3
            feat_val += -((dist < 0) * dist)
        return feat_val

    # -- Proxemics -- #

    def proxemics_features(self, traj):
        """
        Computes distance from end-effector to human proxemics in xy coords.
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar distance sum where
                0: EE is at more than 0.3 meters away from human
                +: EE is closer than 0.3 meters to human
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            waypt = np.hstack((waypt[:7], np.array([0, 0, 0])))
            move_robot(self.env.objectID["robot"], joint_poses=waypt)
            coords = robot_coords(self.env.objectID["robot"])
            EE_coord_xy = coords[6][0:2]
            posH, _ = p.getBasePositionAndOrientation(self.env.objectID["human"])
            human_xy = list(posH[0:2])

            # Modify ellipsis distance.
            EE_coord_xy[1] /= 3
            human_xy[1] /= 3
            dist = np.linalg.norm(EE_coord_xy - human_xy) - 0.3
            feat_val += -((dist < 0) * dist)
        return feat_val

    # -- Coffee -- #
    def coffee_features(self, traj):
        """
        Computes the coffee orientation feature value as the EE orientation.
        ---
        Params:
            traj -- list of waypoints
        Returns:
            dist -- scalar feature sum
        """
        feat_val = np.zeros(1)
        for waypt in traj:
            waypt = np.hstack((waypt[:7], np.array([0, 0, 0])))
            move_robot(self.env.objectID["robot"], joint_poses=waypt)
            orient = robot_orientations(self.env.objectID["robot"])
            EE_orient_x = orient[6][2]
            feat_val += (1 - EE_orient_x)
        return feat_val

    def test_features(self):
        for (idx, feat) in enumerate(self.features):
            steps = 500
            while steps:
                state_info = p.getJointStates(self.env.objectID["robot"], range(11))
                traj = [s[0] for s in state_info[1:]]
                if feat == "table":
                    feat_val = self.table_features(traj)
                elif feat == "human":
                    feat_val = self.human_features(traj)
                elif feat == "laptop":
                    feat_val = self.laptop_features(traj)
                elif feat == "proxemics":
                    feat_val = self.proxemics_features(traj)
                elif feat == "coffee":
                    feat_val = self.coffee_features(traj)
                else:
                    raise NotImplementedError
                print("Feature {} value: {}".format(feat, feat_val))
                steps -= 1
                time.sleep(0.01)
