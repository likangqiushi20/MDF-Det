@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"
"%PYTHON%" -B -u -m hmrn.evaluate_self_test ^
  --checkpoint runs\hmrn\optimized_fair_train_best.pt ^
  --manifest generated\dataset_manifest_local.json ^
  --fixed-grids generated\fair_comparison_grids.json ^
  --truth-cache generated\fair_hmrn_moving_truth.npz ^
  --aoi 01 --first-frame 612 --last-frame 631 ^
  --association-radius 20 --match-radius 9.815537 --progress-every 5 ^
  --output results\hmrn\aoi01_612_631.json
exit /b %errorlevel%
