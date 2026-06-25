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

## Output: `data/gridrobot_<N>.npz`

| Key | Shape | Meaning |
| --- | --- | --- |
| `trajs` | `(N, 19)` | Trajectories: 9 waypoints `[x0, y0, ..., x8, y8]` plus a trailing joint angle. |
| `features` | `(N, 2)` | Per-trajectory features `phi` (scaled): `computer_dist` = summed distance from each waypoint to the grid center; `joint_up` = magnitude of the joint angle. |
| `rewards` | `(N,)` | Ground-truth reward per trajectory, `-theta . phi` (noise-free). |
| `theta` | `(2,)` | Reward weights — one per feature; defines what the human prefers. |
| `beta` | scalar | Rationality / inverse-temperature; higher = less noisy labels. |
| `pref_pairs` | `(P, 2)` | Index pairs `(i, j)` into `trajs` that were compared. |
| `pref_labels` | `(P,)` | Preference per pair: `0` if `trajs[i]` preferred, else `1` (Boltzmann-sampled). |
| `triplets` | `(T, 3)` | Index triplets `(anchor, i, j)` into `trajs`. |
| `triplet_labels` | `(T,)` | Similarity per triplet: `0` if `i` is closer to `anchor` in feature space, else `1`. |

`P` / `T` are set by `--num-prefs` / `--num-triplets` (those keys are omitted if 0).
