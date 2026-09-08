# Revision utilities

- `analysis/paired_aoi_bootstrap.py` reproduces the paired AOI-level bootstrap
  analysis used to quantify uncertainty in the reported MDF-Det improvement.
- `analysis/paired_temporal_block_bootstrap.py` performs the final paired
  temporal-block analysis on per-frame TP/FP/FN, including 10/20/30-frame
  block-size sensitivity and a 95% percentile confidence interval.
- `figures/make_spgf_probability_figure.py` creates the six-AOI input/prior
  figure.
- `figures/make_spgf_cross_dataset_figure.py` creates the combined WPAFB 2009
  and zero-shot Greene 2007 visualization.
- `figures/visualize_spgf_training_coverage.py` audits and visualizes the
  training, validation, and held-out test regions.

All utilities accept explicit input/output paths or document their expected
repository-relative layout. Generated figures and statistics belong under
`result/`, which is ignored by Git.
