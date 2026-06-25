import copy
import random
from enum import IntEnum

import numpy as np
from matplotlib import pyplot as plt


class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# Change in (x, y) produced by each action.
ACTION_DELTAS = {
    Actions.UP: (0, -1),
    Actions.DOWN: (0, 1),
    Actions.LEFT: (-1, 0),
    Actions.RIGHT: (1, 0),
}


class Gridworld:
    """
    An X by Y gridworld MDP for an agent that moves between a start and a goal
    position while avoiding rectangular obstacles.

    A trajectory is stored as a flat list of waypoints:
        [x0, y0, x1, y1, ..., xT, yT]
    """
    Actions = Actions

    def __init__(self, X, Y, obstacles, starts, goals, verbose=True):
        self.X = X
        self.Y = Y
        self.S = X * Y
        self.A = len(Actions)

        # Each obstacle is [[x_lo, y_hi], [x_hi, y_lo]] (an axis-aligned box).
        self.obstacles = obstacles
        self.starts = starts
        self.goals = goals

        if verbose:
            print("------ Agent MDP ------")
            print("num X : ", self.X)
            print("num Y : ", self.Y)
            print("obstacles : ", len(self.obstacles) if self.obstacles else 0)
            print("-----------------------")

    def generate_trajs(self, samples):
        """
        Enumerate every shortest-path (length T = distance + 1) trajectory for
        each (start, goal) pair, then randomly sub-sample down to `samples`.
        """
        assert len(self.starts) == len(self.goals), "Need same number of start and goal tuples."
        trajs = []

        def recurse_actions(curr, timestep):
            # Recursively build legal action combinations that reach the goal.
            if timestep == T - 1:
                if curr[0] == goal[0] and curr[1] == goal[1]:
                    trajs.append(list(traj))
                return
            # Prune branches that can no longer reach the goal in time.
            if self.distance(curr, goal) > T - 1 - timestep:
                return
            rand_actions = list(Actions)
            random.shuffle(rand_actions)
            for a in rand_actions:
                s_prime, illegal = self.transition_helper(curr, a)
                if not illegal:
                    traj[2 * (timestep + 1)] = s_prime[0]
                    traj[2 * (timestep + 1) + 1] = s_prime[1]
                    recurse_actions(s_prime, timestep + 1)

        for start, goal in zip(self.starts, self.goals):
            T = self.distance(start, goal) + 1
            traj = list(start) + [None] * (2 * (T - 1))
            recurse_actions(start, 0)

        trajs = copy.deepcopy(trajs)
        samples = min(samples, len(trajs))
        return random.sample(trajs, samples)

    def transition_helper(self, s, a):
        """Apply action `a` at state `s`; return (next_state, illegal)."""
        x, y = s
        assert 0 <= a < self.A, "undefined action {}".format(a)

        dx, dy = ACTION_DELTAS[Actions(a)]
        x_prime, y_prime = x + dx, y + dy

        illegal = False
        if x_prime < 0 or x_prime >= self.X or y_prime < 0 or y_prime >= self.Y:
            illegal = True
            s_prime = s
        else:
            s_prime = (x_prime, y_prime)
            if self.is_blocked(s_prime):
                illegal = True
        if self.is_blocked(s):
            illegal = True
        return s_prime, illegal

    def distance(self, s1, s2):
        # Manhattan distance.
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def is_blocked(self, s):
        """Return True if state `s` falls inside any obstacle box."""
        if not self.obstacles:
            return False
        x, y = s
        for box in self.obstacles:
            if box[0][0] <= x <= box[1][0] and box[1][1] <= y <= box[0][1]:
                return True
        return False

    def visualize(self, trajs, rews, im_path=None, max_display=20):
        """Plot the lowest-reward trajectories on top of the obstacle map."""
        trajs = np.asarray(trajs)
        rews = np.asarray(rews)

        world = 0.5 * np.ones((self.Y, self.X))
        for obstacle in (self.obstacles or []):
            lower, upper = obstacle[0], obstacle[1]
            world[upper[1]:lower[1] + 1, lower[0]:upper[0] + 1] = 1.0
        plt.imshow(world, cmap='Greys', interpolation='nearest')

        sorted_indices = np.argsort(rews)[:max_display]
        for traj in trajs[sorted_indices]:
            plt.scatter(traj[0], traj[1], color="orange", marker="o", s=100)
            plt.scatter(traj[-2], traj[-1], color="orange", marker="x", s=300)
            traj_x = traj[0::2]
            traj_y = traj[1::2]
            plt.plot(traj_x, traj_y, c="r", linewidth=3, alpha=max(1 / max_display, 0.05))

        plt.grid(alpha=0.3)
        if im_path is not None:
            plt.savefig(im_path)
        else:
            plt.show()
        plt.close()
