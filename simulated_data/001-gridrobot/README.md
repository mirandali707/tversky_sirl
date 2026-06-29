# mini-gridrobot

Generates synthetic gridrobot trajectories with simulated "human" labels. Like
`mini-gridworld`, but each trajectory also carries an end-effector **joint angle**,
making the input 19-dimensional.

**unlike the SIRL paper, there are no obstacles (other than an implied laptop at the center /origin), so the feature vector is 2-dimensional. strangely, this is inconsistent with the text and figure 2, but consistent with figure 3, and nowhere in the codebase can I find mention of these blue / green obstacles... so maybe this was the GridRobot data they used after all?**

```bash
python generate_traj_sets.py --samples 1960 --num-prefs 500 --num-triplets 200 --visualize
```

A 5x5 obstacle-free world; the four corners are starts heading to the opposite
corner, enumerated across 7 discrete joint angles (-90 deg to +90 deg). The human
scores a trajectory by `reward = -theta . phi`, where `phi = [computer_dist,
joint_up]`, and answers queries Boltzmann-rationally with inverse-temperature
`beta`.

The config holds a nested list of reward weights `thetas` (one `theta` per row).
Trajectories and `pref_pairs` are shared across all of them, but rewards and
preference labels are generated separately per `theta` — so the bundle carries
one `theta_<i>` / `rewards_<i>` / `pref_labels_<i>` set per row.

## Output: `data/gridrobot_<N>.npz`

| Key | Shape | Meaning |
| --- | --- | --- |
| `trajs` | `(N, 19)` | Trajectories: 9 waypoints `[x0, y0, ..., x8, y8]` plus a trailing joint angle. |
| `features` | `(N, 2)` | Per-trajectory features `phi` (scaled): `computer_dist` = summed distance from each waypoint to the grid center; `joint_up` = magnitude of the joint angle. |
| `theta_<i>` | `(2,)` | Reward weights for theta `i` — one per feature; defines what that human prefers. One per row of `thetas`. |
| `rewards_<i>` | `(N,)` | Ground-truth reward per trajectory under `theta_<i>`, `-theta_<i> . phi` (noise-free). |
| `beta` | scalar | Rationality / inverse-temperature; higher = less noisy labels. |
| `pref_pairs` | `(P, 2)` | Index pairs `(i, j)` into `trajs` that were compared — shared across all thetas. |
| `pref_labels_<i>` | `(P,)` | Preference per pair under `theta_<i>`: `0` if `trajs[i]` preferred, else `1` (Boltzmann-sampled). |
| `triplets` | `(T, 3)` | Index triplets `(anchor, i, j)` into `trajs`. |
| `triplet_labels` | `(T,)` | Similarity per triplet: `0` if `i` is closer to `anchor` in feature space, else `1`. |

`P` / `T` are set by `--num-prefs` / `--num-triplets` (those keys are omitted if 0).
