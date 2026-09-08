# ClusterNet + FoveaNet

This is the consolidated default ClusterNet method. It contains the trained ClusterNet proposer and the improved FoveaNet stage in one package and one checkpoint.

Changes:

- Fovea training samples come from actual frozen ClusterNet ROOBIs.
- Positive, hard-negative and random-negative samples are stored at a 1:3 ratio.
- The bank stores coordinates only. Existing uint8 frame arrays remain memory-mapped, so no second multi-gigabyte image copy is created.
- Positive crops receive up to 16-pixel jitter.
- The network predicts a full 64x64 heatmap for each 128x128x5 input.
- Training uses focal BCE plus Dice instead of sparse-map MSE.
- GPU AMP, frame-grouped batches, same-frame RAM caching, checkpoint resume and early stopping are enabled.
- Full ROOBIs are covered with overlapping patches and Hann-weighted average stitching.
- Detection uses absolute probability thresholds plus local-maximum NMS.

From the public `baselines` directory, run in order:

1. `clusternet\run_build_bank.cmd`
2. `clusternet\run_train.cmd`
3. `clusternet\run_evaluate.cmd`

The bank builder is resumable: completed frame/AOI shards are skipped. The trainer writes `latest.pt` every epoch and `best.pt` whenever validation patch F1 improves. Re-running the training CMD resumes automatically.

Smoke verification on AOI01 frames 550-552/561-562 produced an exact 1:3 positive/negative bank. After disabling inappropriate variable-batch cuDNN exhaustive benchmarking, one smoke epoch dropped from about 35 seconds to about 1.45 seconds. Ten smoke epochs completed in about 9 seconds. This smoke model is only a pipeline check and is not a benchmark model.
