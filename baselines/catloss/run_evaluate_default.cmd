@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"
"%PYTHON%" -B -u -m catloss.evaluate_self_test ^
  --checkpoint runs\catloss\optimized_fair_train_best.pt ^
  --manifest generated\dataset_manifest_local.json ^
  --fixed-grids generated\fair_comparison_grids.json ^
  --truth-cache generated\fair_catloss_moving_truth.npz ^
  --aoi 01 --first-frame 612 --last-frame 631 ^
  --match-radius 9.815537 --batch-size 512 --progress-every 5 ^
  --output results\catloss\aoi01_612_631.json
exit /b %errorlevel%
