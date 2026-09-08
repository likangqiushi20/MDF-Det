# Revised SPGF scene-prior module

SPGF estimates two dense probabilities from native-resolution grayscale scene
context: general vehicle accessibility and moving-vehicle occurrence. Their
geometric mean forms the scene prior used by MDF-Det.

The network is fully convolutional and has no coordinate input. Its training
builder applies a strict geographic hold-out: all image patches and labels that
touch AOI01, AOI02, AOI03, AOI34, AOI40, or AOI41 (plus a safety margin) are
excluded. This prevents the evaluated AOI locations from entering supervision.

## Input conventions

- Frames: `frame000100.png`, `frame000101.png`, ... in one directory.
- Truth CSV columns: `id`, `FRAME_NUMBER`, `LATITUDE`, `LONGITUDE`, `X`, `Y`.
- Homography NPZ keys: `frame_numbers`, `matrices`, `valid`, `image_sizes`.
  Each valid matrix maps one frame to the following frame.

## 1. Build audited training shards

```bash
python -m SPGF.build_dataset \
  --frames /path/to/full_frame_png \
  --homographies /path/to/adjacent_homographies.npz \
  --truth-csv /path/to/truth.csv \
  --output data/spgf_v4
```

Default training/validation ranges are 100--379 and 400--499. The default input
patch is 384 x 384, the trajectory-label history is 20 frames, the moving
threshold is 0.8 m, and the AOI exclusion safety margin is 64 pixels. Existing
completed shards are reused unless `--overwrite` is supplied. Training refuses
to start unless `dataset_summary.json` confirms that all shard audits passed.

## 2. Train or resume

```bash
python -m SPGF.train \
  --data data/spgf_v4 \
  --output models/spgf_v4 \
  --epochs 80 \
  --batch-size 4
```

Resume from a checkpoint with:

```bash
python -m SPGF.train \
  --data data/spgf_v4 \
  --output models/spgf_v4 \
  --epochs 80 \
  --resume models/spgf_v4/latest_semantic_prior_v4.h5
```

## 3. Build one fixed prior for an AOI

```bash
python -m SPGF.infer \
  --frames /path/to/fixed_aoi_png \
  --model models/spgf_v4/best_semantic_prior_v4.h5 \
  --output result/aoi01_prior \
  --context-start 100 \
  --context-stop 119 \
  --mode tiled \
  --tile 384 \
  --overlap 96
```

The detector consumes `combined_prior.npy`. Because the AOI grid is fixed, the
same prior is reused for its continuous frame sequence.

## 4. Optional GT-free topology cleanup

`postprocess.py` connects short directional gaps and suppresses isolated compact
responses using only the predicted prior. It does not read target-AOI truth.

```bash
python -m SPGF.postprocess \
  --source result/raw_priors \
  --output result/topology_priors
```

Use the topology step only when the source directory follows the six-AOI layout
expected by the script. All thresholds are command-line options.

## 5. Zero-shot Greene 2007 visualization

```bash
python -m SPGF.zero_shot_greene2007 \
  --source /path/to/Greene2007-DataSet-Disc1 \
  --model models/spgf_v4/best_semantic_prior_v4.h5 \
  --output result/greene2007_zero_shot
```

No Greene labels or fine-tuning are used by this utility.
