`005-sirl-tversky`
- [ ] try giving TverskySim (anchor, positive) and (anchor, negative) instead of (hi, lo) (e.g. just max, min pairs)
- [ ] train SIRL -> freeze SIRL -> train TverskySim
- [ ] train SIRL, TverskySim at the same time?
- [ ] alternate???
- [ ] baselines:
	- [ ] random SIRL -> random TverskySim
	- [ ] train SIRL -> random TverskySim
	- [ ] random SIRL -> train TverskySim
eval:
- [ ] t-SNE with ground truth feats (precomputed distance matrix for TS)
- [ ] query-ability of TS
- [ ] FPE, TPA of SIRL
