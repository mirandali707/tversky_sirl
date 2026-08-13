# `005-sirl-tversky`
we have shown (in expt 004) that SIRL input -> training Tversky Sim on (hi, lo) pairs (with InfoNCE loss) does better than raw traj -> random init Tversky Sim, which is a bit apples to oranges but a hopeful sign:

raw trajectory -> random init TverskySim, no training (from expt 004)
![](../004-infonce-tversky-sim/figs/random_fbank_size.png)

SIRL -> TverskySim (trained with InfoNCE loss) (from expt 004)
![](../004-infonce-tversky-sim/figs/sirl_ljts_fbank_size.png)

... so, in this experiment we try to make this comparison more apples to apples, and resolve several bugs / add some new methods along the way.

# methods
* `random.yaml` / `random_no_sirl_ts`: raw trajectory -> random init TverskySim (no training)
![](figs/random_no_sirl_ts_fbank_size.png)
	* at high feature bank sizes, structure seems to approach that of raw data / PCA:
	![](results/random_no_sirl_ts/random_no_sirl_ts_1786628652_tsne_laptop.png)
* `baseline.yaml` / `no_sirl_ts`: raw trajectory -> train TverskySim
![](figs/no_sirl_ts_fbank_size.png)
	* seems to be learning something, drawing all e.g. high laptop points together, somewhat noisily; bad on upright
	![](results/no_sirl_ts/no_sirl_ts_1786627879_tsne_laptop.png)
* `frozen_sirl_ts.yaml` / `frozen_sirl_ts`: raw trajectory -> SIRL (checkpoint, frozen) -> train TverskySim
![](figs/frozen_sirl_ts_fbank_size.png)
	* more successfully draws high laptop / upright points close together
	![](results/frozen_sirl_ts/frozen_sirl_ts_1786635013_tsne_laptop.png)
	![](results/frozen_sirl_ts/frozen_sirl_ts_1786635013_tsne_upright.png)
	* accuracy plateaus around 70% (s_ap > s_an)
* `sirl_ts.yaml` / `sirl_ts`: raw trajectory -> SIRL -> TverskySim end-to-end, all unfrozen
	![](figs/sirl_ts_fbank_size.png)
	* accuracy gets to around 90% (s_ap > s_an) and seems to not have plateaued
	* but, learned similarity has outliers, doesn't seem as well-conditioned as `frozen_sirl_ts`
	![](results/sirl_ts/sirl_ts_1786639576_tsne_laptop.png)

compare to tversky proj encoder, regular triplet loss (notably did not use "ratio" + "normalize"):
![](../003-tversky-methods-all-eval/figs/tversky_proj_fbank_size.png)

# new changes
* we train TverskySim on (a,p,n) triplets with triplet loss instead of just (hi, lo) pairs with InfoNCE loss so that we can use the whole dataset and use the same data as SIRL
* TverskySim uses "ratio" instead of "contrast" and normalize=True to produce similarity values bounded between 0 and 1
* triplet loss now uses Pytorch's TripletMarginWithDistanceLoss (again), with distance = 1 - similarity
* alpha, beta are not learned directly, because alpha was being pushed to be negative - we learn `raw_alpha` and `raw_beta` and pass them through a softplus to get alpha, beta for the TverskySim similarity computation
* added new eval method: use TverskySimilarity to compute all-pairs distance matrix and pass this into t-SNE with `metric="precomputed"` to get a vibe for whether or not TverskySimilarity is learning some structure.
	* manually starting one kaleido server per run and also turning gradients off for the all-pairs similarity computation sped things up a lot