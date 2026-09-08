# Reproduced WAMI baselines

This directory contains the independently reproduced baseline implementations
used for the MDF-Det comparison. They are research reproductions based on the
corresponding papers and are not the original authors' official repositories.

## Included methods

| Method | Directory | Main idea | Training entry |
|---|---|---|---|
| ClusterNet + FoveaNet | `clusternet/` | Low-resolution cluster proposals followed by high-resolution local refinement | `python -m clusternet.train --help` |
| HMRN | `hmrn/` | Center and displacement prediction with temporal heatmap feedback | `python -m hmrn.train --help` |
| HM-Net | `hm_net/` | Multi-branch encoding, heatmap reconstruction and temporal association | `python -m hm_net.train --help` |
| CATLoss | `catloss/` | Objectness classification and crowd-aware localization | `python -m catloss.train --help` |

The `common/` package provides the shared WPAFB manifest, fixed geographic
grids, truth projection and one-to-one point-matching utilities. TTE-KH is not
included in this public release.

## Installation

Install the main MDF-Det requirements first, then the PyTorch/geospatial
dependencies:

```bash
pip install -r baselines/requirements.txt
```

For GPU execution, install a PyTorch build compatible with the local CUDA
driver by following the official PyTorch installation selector.

## Data and generated files

The implementations expect the WPAFB imagery and truth files to be supplied
locally. Dataset paths, cached frame arrays, projected truth files, proposal
banks, checkpoints and evaluation outputs are deliberately excluded from Git.
Generate them locally under `baselines/generated/` and `baselines/runs/` or
override those locations in the configuration files.

The shared fair-comparison configuration is
`common/configs/fair_comparison.json`. Each method's paper and optimized
settings are under its `configs/` directory. Threshold selection must use only
the training/validation split; SELF-TEST data should be reserved for final
evaluation.

## Pretrained weights

Weights are not stored in this Git repository. The separately prepared model
bundle uses the following layout:

```text
MDF-Det/
Dual-CNN/
baselines/ClusterNet/
baselines/HMRN/
baselines/HM-Net/
baselines/CATLoss/
```

After publishing the bundle on Hugging Face, place the downloaded checkpoints
at the paths specified by each inference configuration or update the
`checkpoint` field accordingly.

## Reproducibility note

These methods share identical AOI extents, source resolution, truth projection
and one-to-one matching code for the reported comparison. Method-specific
architectures and objectives remain separated in their own packages.
