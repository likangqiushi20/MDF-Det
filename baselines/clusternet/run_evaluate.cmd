@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"

"%PYTHON%" -B -u -m clusternet.evaluate_self_test ^
  --external-repo "%CD%" ^
  --checkpoint runs\clusternet\best.pt ^
  --manifest generated\dataset_manifest_local.json ^
  --fixed-grids generated\fair_comparison_grids.json ^
  --truth-cache generated\fair_clusternet_truth.npz ^
  --split self_test ^
  --aoi 01 ^
  --first-frame 612 ^
  --last-frame 631 ^
  --cluster-quantile 0.90 ^
  --thresholds 0.45 ^
  --nms-radius 4 ^
  --match-radius 9.815537 ^
  --batch-size 64 ^
  --output results\clusternet\aoi01_612_631.json
exit /b %errorlevel%
