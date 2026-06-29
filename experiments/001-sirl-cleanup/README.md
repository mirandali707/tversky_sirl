uses data from `simulated_data/001-gridrobot` to evaluate FPE, TPA of:
* PCA
* random baseline (untrained SIRL)
* SIRL
* Tversky SIRL (TverskySimilarity in triplet loss, MLP left as-is)
* Tversky SIRL 2 (TverskyProjection instead of MLP, TverskySimilarity in triplet loss) - needs a better name I'm sorry