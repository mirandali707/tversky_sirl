import copy
import random

import numpy as np
from matplotlib import pyplot as plt

from gridworld import Gridworld, Actions


class Gridrobot(Gridworld):
    """
    A Gridworld whose agent additionally carries a discrete end-effector joint
    angle. The angle is constant along a trajectory and is appended as the final
    element, so a trajectory is 19-dimensional:

        [x0, y0, x1, y1, ..., xT, yT, joint_angle]
    """

    # Discrete joint angles (-90 deg .. +90 deg in 30 deg steps) and the color
    # used to draw trajectories with that angle.
    joint_dict = {-1.5708: "blue",
                  -1.0472: "green",
                  -0.5236: "green",
                  0.0: "yellow",
                  0.5236: "orange",
                  1.0472: "orange",
                  1.5708: "red"}

    def generate_trajs(self, samples):
        """
        For every joint angle, enumerate all shortest-path trajectories for each
        (start, goal) pair, append the joint angle, then sub-sample to `samples`.
        """
        assert len(self.starts) == len(self.goals), "Need same number of start and goal tuples."
        trajs = []

        def recurse_actions(curr, timestep, joint_angle):
            if timestep == T - 1:
                if curr[0] == goal[0] and curr[1] == goal[1]:
                    trajs.append(list(traj) + [joint_angle])
                return
            if self.distance(curr, goal) > T - 1 - timestep:
                return
            rand_actions = list(Actions)
            random.shuffle(rand_actions)
            for a in rand_actions:
                s_prime, illegal = self.transition_helper(curr, a)
                if not illegal:
                    traj[2 * (timestep + 1)] = s_prime[0]
                    traj[2 * (timestep + 1) + 1] = s_prime[1]
                    recurse_actions(s_prime, timestep + 1, joint_angle)

        for joint_angle in self.joint_dict.keys():
            for start, goal in zip(self.starts, self.goals):
                T = self.distance(start, goal) + 1
                traj = list(start) + [None] * (2 * (T - 1))
                recurse_actions(start, 0, joint_angle)

        trajs = copy.deepcopy(trajs)
        samples = min(samples, len(trajs))
        return random.sample(trajs, samples)

    def visualize(self, trajs, rews, im_path=None, max_display=20):
        """Plot the highest-reward trajectories, colored by their joint angle."""
        trajs = np.asarray(trajs)
        rews = np.asarray(rews)

        world = 0.5 * np.ones((self.Y, self.X))
        for obstacle in (self.obstacles or []):
            lower, upper = obstacle[0], obstacle[1]
            world[upper[1]:lower[1] + 1, lower[0]:upper[0] + 1] = 1.0
        plt.imshow(world, cmap='Greys', interpolation='nearest')

        sorted_indices = np.argsort(rews)[-max_display:]
        for traj in trajs[sorted_indices]:
            joint = traj[-1]
            path = traj[:-1]
            plt.scatter(path[0], path[1], color="orange", marker="o", s=100)
            plt.scatter(path[-2], path[-1], color="orange", marker="x", s=300)
            traj_x = path[0::2]
            traj_y = path[1::2]
            plt.plot(traj_x, traj_y, c=self.joint_dict[joint], linewidth=3,
                     alpha=max(1 / max_display, 0.05))

        plt.grid(alpha=0.3)
        if im_path is not None:
            plt.savefig(im_path)
        else:
            plt.show()
        plt.close()
