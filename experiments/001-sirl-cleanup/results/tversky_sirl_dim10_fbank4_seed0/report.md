# Tversky feature report: `tversky_sirl_dim10_fbank4_seed0.pth`

config: `../configs/tversky_sirl.yaml`  
trajectories: (1960, 19), features: (1960, 2), feature bank: (4, 10)

# sample similar + dissimilar trajectories

## query traj `1729` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 0.0000, salience = 0.0645

![traj 1729](figs/query_traj1729.png)

### 3 most similar

- traj `1583` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 0.3333, salience = 0.0630 (sim = 0.0033)

![traj 1583](figs/q1729_most_traj1583.png)

- traj `1463` — features: laptop (computer_dist) = 0.2185, upright (joint_up) = 1.0000, salience = 0.0668 (sim = 0.0031)

![traj 1463](figs/q1729_most_traj1463.png)

- traj `1658` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 1.0000, salience = 0.0675 (sim = 0.0028)

![traj 1658](figs/q1729_most_traj1658.png)

### 3 least similar

- traj `216` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 1.0000, salience = 8.1493 (sim = -3.7186)

![traj 216](figs/q1729_least_traj216.png)

- traj `1827` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 1.0000, salience = 8.0475 (sim = -3.6750)

![traj 1827](figs/q1729_least_traj1827.png)

- traj `1533` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 0.6667, salience = 8.0387 (sim = -3.6678)

![traj 1533](figs/q1729_least_traj1533.png)

## query traj `788` — features: laptop (computer_dist) = 0.5000, upright (joint_up) = 0.6667, salience = 0.0000

![traj 788](figs/query_traj788.png)

### 3 most similar

- traj `874` — features: laptop (computer_dist) = 0.2185, upright (joint_up) = 1.0000, salience = 0.0000 (sim = 0.0000)

![traj 874](figs/q788_most_traj874.png)

- traj `360` — features: laptop (computer_dist) = 0.0905, upright (joint_up) = 0.3333, salience = 0.0000 (sim = 0.0000)

![traj 360](figs/q788_most_traj360.png)

- traj `1516` — features: laptop (computer_dist) = 0.5000, upright (joint_up) = 1.0000, salience = 0.0000 (sim = 0.0000)

![traj 1516](figs/q788_most_traj1516.png)

### 3 least similar

- traj `216` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 1.0000, salience = 8.1493 (sim = -3.8789)

![traj 216](figs/q788_least_traj216.png)

- traj `1827` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 1.0000, salience = 8.0475 (sim = -3.8305)

![traj 1827](figs/q788_least_traj1827.png)

- traj `1533` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 0.6667, salience = 8.0387 (sim = -3.8263)

![traj 1533](figs/q788_least_traj1533.png)


# sort trajectories by salience

10 trajectories sampled evenly from least to most salient.

## salience rank 1/10: traj `638` — features: laptop (computer_dist) = 0.3995, upright (joint_up) = 1.0000, salience = 0.0000

![traj 638](figs/salience_rank1_traj638.png)

## salience rank 2/10: traj `1314` — features: laptop (computer_dist) = 0.3995, upright (joint_up) = 1.0000, salience = 0.0000

![traj 1314](figs/salience_rank2_traj1314.png)

## salience rank 3/10: traj `1505` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 1.0000, salience = 0.0000

![traj 1505](figs/salience_rank3_traj1505.png)

## salience rank 4/10: traj `1787` — features: laptop (computer_dist) = 0.1810, upright (joint_up) = 0.0000, salience = 0.0565

![traj 1787](figs/salience_rank4_traj1787.png)

## salience rank 5/10: traj `1736` — features: laptop (computer_dist) = 0.0905, upright (joint_up) = 0.3333, salience = 0.1900

![traj 1736](figs/salience_rank5_traj1736.png)

## salience rank 6/10: traj `1677` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 0.0000, salience = 0.3303

![traj 1677](figs/salience_rank6_traj1677.png)

## salience rank 7/10: traj `696` — features: laptop (computer_dist) = 0.0905, upright (joint_up) = 1.0000, salience = 0.5168

![traj 696](figs/salience_rank7_traj696.png)

## salience rank 8/10: traj `237` — features: laptop (computer_dist) = 0.2185, upright (joint_up) = 1.0000, salience = 0.7409

![traj 237](figs/salience_rank8_traj237.png)

## salience rank 9/10: traj `538` — features: laptop (computer_dist) = 0.3090, upright (joint_up) = 0.3333, salience = 1.4325

![traj 538](figs/salience_rank9_traj538.png)

## salience rank 10/10: traj `1235` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 0.3333, salience = 3.4634

![traj 1235](figs/salience_rank10_traj1235.png)


# max feature value - min feature value t-test of means

## laptop (computer_dist): max - min

expression `s(29)-s(1)` → 0 Tversky feature(s) in the difference set

**traj `29` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 0.6667, salience = 0.0000**

![traj 29](figs/feat0_max-min_a_traj29.png)

minus

**traj `1` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 0.3333, salience = 0.2521**

![traj 1](figs/feat0_max-min_b_traj1.png)

equals top instances:

*(empty feature set — no instances retrieved)*

## laptop (computer_dist): min - max

expression `s(1)-s(29)` → 1 Tversky feature(s) in the difference set

**traj `1` — features: laptop (computer_dist) = 0.0000, upright (joint_up) = 0.3333, salience = 0.2521**

![traj 1](figs/feat0_min-max_a_traj1.png)

minus

**traj `29` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 0.6667, salience = 0.0000**

![traj 29](figs/feat0_min-max_b_traj29.png)

equals top instances:

- traj `216` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 1.0000, salience = 8.1493 (measure = 2.0606, salience = 2.0606)

![traj 216](figs/feat0_min-max_top_traj216.png)

- traj `1533` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 0.6667, salience = 8.0387 (measure = 2.0319, salience = 2.0319)

![traj 1533](figs/feat0_min-max_top_traj1533.png)

- traj `778` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 0.3333, salience = 7.8414 (measure = 1.9838, salience = 1.9838)

![traj 778](figs/feat0_min-max_top_traj778.png)

- traj `1827` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 1.0000, salience = 8.0475 (measure = 1.9833, salience = 1.9833)

![traj 1827](figs/feat0_min-max_top_traj1827.png)

- traj `1382` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 0.6667, salience = 7.9834 (measure = 1.9661, salience = 1.9661)

![traj 1382](figs/feat0_min-max_top_traj1382.png)

- traj `340` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 0.3333, salience = 7.8496 (measure = 1.9307, salience = 1.9307)

![traj 340](figs/feat0_min-max_top_traj340.png)

- traj `1284` — features: laptop (computer_dist) = 0.5905, upright (joint_up) = 0.0000, salience = 7.5475 (measure = 1.9133, salience = 1.9133)

![traj 1284](figs/feat0_min-max_top_traj1284.png)

- traj `439` — features: laptop (computer_dist) = 0.3995, upright (joint_up) = 1.0000, salience = 7.5185 (measure = 1.8880, salience = 1.8880)

![traj 439](figs/feat0_min-max_top_traj439.png)

- traj `1944` — features: laptop (computer_dist) = 1.0000, upright (joint_up) = 0.0000, salience = 7.6715 (measure = 1.8857, salience = 1.8857)

![traj 1944](figs/feat0_min-max_top_traj1944.png)

- traj `948` — features: laptop (computer_dist) = 0.7815, upright (joint_up) = 1.0000, salience = 7.6484 (measure = 1.8729, salience = 1.8729)

![traj 948](figs/feat0_min-max_top_traj948.png)

## upright (joint_up): max - min

expression `s(2)-s(3)` → 0 Tversky feature(s) in the difference set

**traj `2` — features: laptop (computer_dist) = 0.0905, upright (joint_up) = 1.0000, salience = 0.0000**

![traj 2](figs/feat1_max-min_a_traj2.png)

minus

**traj `3` — features: laptop (computer_dist) = 0.5000, upright (joint_up) = 0.0000, salience = 0.0000**

![traj 3](figs/feat1_max-min_b_traj3.png)

equals top instances:

*(empty feature set — no instances retrieved)*

## upright (joint_up): min - max

expression `s(3)-s(2)` → 0 Tversky feature(s) in the difference set

**traj `3` — features: laptop (computer_dist) = 0.5000, upright (joint_up) = 0.0000, salience = 0.0000**

![traj 3](figs/feat1_min-max_a_traj3.png)

minus

**traj `2` — features: laptop (computer_dist) = 0.0905, upright (joint_up) = 1.0000, salience = 0.0000**

![traj 2](figs/feat1_min-max_b_traj2.png)

equals top instances:

*(empty feature set — no instances retrieved)*


## t-test of means

- **laptop (computer_dist)**: skipped — at least one feature set is empty (max-min: 0 instances, min-max: 10 instances)

- **upright (joint_up)**: skipped — at least one feature set is empty (max-min: 0 instances, min-max: 0 instances)

*Can't run any t-tests on this model — the feature sets are empty.*

