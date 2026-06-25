# mini-gridworld

Generates synthetic gridworld trajectories with simulated "human" labels.

```bash
python generate_traj_sets.py --samples 120 --num-prefs 500 --num-triplets 200 --visualize
```

**unlike the SIRL paper, there is no joint angle, so the trajectories are 18-dimensional.**

A 9x9 world with one obstacle near each corner; all four corners are starts
heading to the center. The human scores a trajectory by `reward = -theta . phi`,
where `phi` is the obstacle-distance feature vector, and answers queries
Boltzmann-rationally with inverse-temperature `beta`.

## Output: `data/gridworld_<N>.npz`

| Key | Shape | Meaning |
| --- | --- | --- |
| `trajs` | `(N, 18)` | Trajectories: 9 waypoints flattened as `[x0, y0, ..., x8, y8]`. |
| `features` | `(N, 4)` | Per-trajectory features `phi`: summed distance to each of the 4 obstacles (scaled). |
| `rewards` | `(N,)` | Ground-truth reward per trajectory, `-theta . phi` (noise-free). |
| `theta` | `(4,)` | Reward weights — one per obstacle feature; defines what the human prefers. |
| `beta` | scalar | Rationality / inverse-temperature; higher = less noisy labels. |
| `pref_pairs` | `(P, 2)` | Index pairs `(i, j)` into `trajs` that were compared. |
| `pref_labels` | `(P,)` | Preference per pair: `0` if `trajs[i]` preferred, else `1` (Boltzmann-sampled). |
| `triplets` | `(T, 3)` | Index triplets `(anchor, i, j)` into `trajs`. |
| `triplet_labels` | `(T,)` | Similarity per triplet: `0` if `i` is closer to `anchor` in feature space, else `1`. |

`P` / `T` are set by `--num-prefs` / `--num-triplets` (those keys are omitted if 0).
