# MDF-Det

MDF-Det detects and tracks small moving vehicles in Wide Area Motion Imagery
(WAMI). It combines image registration, multi-frame background subtraction,
CNN candidate refinement, position regression, optional optical-flow motion
fusion, and Kalman-filter tracking.

The repository contains source code and small normalization parameters only.
Model weights, datasets, cached files, and experiment outputs are intentionally
excluded.

## Pipeline

1. Register previous frames to the current frame with ORB features and a
   homography.
2. Build a temporal median background from registered frames.
3. Generate moving-object candidates by background subtraction.
4. Reject false candidates with a binary CNN.
5. Estimate vehicle centers with a regression CNN.
6. Optionally fuse optical-flow motion, scene priors, and Kalman tracking.

## Repository structure

```text
MovingObjectDetector/       Registration, background modeling and refinement
TrainNetwork/               Legacy CNN training and model-loading utilities
SimpleTracker/              Kalman filter implementation
WAMI_detector.py            Basic full-frame detector
WAMI_detector_multi_AOI.py  Multi-AOI detector with motion/prior fusion
WAMI_detector_multi_AOI_origreg.py
train_*.py                  Attention/regression/scene-prior training scripts
compute_metrics*.py         Evaluation utilities
```

## Installation

Python 3.9 is recommended for the legacy TensorFlow and NumPy APIs used by this
code.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the NITF/AOI and OpenStreetMap utilities, install GDAL for your platform
and then run:

```bash
pip install -r requirements-geo.txt
```

## Model files

Inference requires model weights that are not stored in Git. Place your
weights at the following paths:

```text
Models/BinaryClassification/saved_model_2.model
Models/Regression/saved_model_3.model
regression_spatial_attention.h5        # multi-AOI attention regression
scene_prior_net.h5                     # multi-AOI scene prior
```

The corresponding small normalization files are included:

```text
Models/BinaryClassification/saved_image_norm_2.model
Models/Regression/saved_image_norm_3.model
regression_norm_params.npz
```

If you do not already have compatible weights, use the training scripts in
`TrainNetwork/` and the top-level `train_*.py` files to create them. Training
data is not included.

## Basic detector

Prepare a directory containing chronologically named PNG/JPEG frames, then run:

```bash
python WAMI_detector.py \
  --InputFolder /path/to/frames \
  --OutputFolder ./WAMI-output \
  --NNModelFolder ./Models \
  --BSThreshold 8 \
  --NumOfTemplate 3
```

Processed frames are written to the output directory and detection coordinates
are written as CSV files under `WAMI-output/CSV/`.

## Multi-AOI detector

The multi-AOI entry point additionally expects extracted AOI frames, a truth
CSV file containing the geospatial metadata used by the script, and the
attention-regression and scene-prior weights:

```bash
python WAMI_detector_multi_AOI.py \
  --png_root /path/to/aoi_frames \
  --truth_csv /path/to/truth.csv \
  --output_base ./WAMI-output \
  --binary_model_dir ./Models \
  --regression_model ./regression_spatial_attention.h5 \
  --regression_norm ./regression_norm_params.npz \
  --prior_model ./scene_prior_net.h5
```

Use `python WAMI_detector_multi_AOI.py --help` for all AOI, threshold, motion,
and matching options.

## Data and weights policy

Do not commit model weights, WAMI imagery, generated labels, or experiment
outputs. The supplied `.gitignore` excludes these files. For reproducible
inference, publish weights separately (for example, as a GitHub Release or an
external download) and document their checksums here.

## Reference

Y. Zhou and S. Maskell, "Detecting and Tracking Small Moving Objects in Wide
Area Motion Imagery (WAMI) Using Convolutional Neural Networks (CNNs)," 2019
22nd International Conference on Information Fusion (FUSION), pp. 1-8,
doi: 10.23919/FUSION43075.2019.9011271.

## License

See [LICENSE](LICENSE).
