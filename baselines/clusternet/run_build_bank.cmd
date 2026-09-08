@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"

"%PYTHON%" -B -u -m clusternet.build_roobi_bank ^
  --external-repo "%CD%" ^
  --cluster-checkpoint runs\clusternet\best.pt ^
  --frame-cache generated\frame_cache\cache.json ^
  --truth-cache generated\fair_clusternet_truth.npz ^
  --output generated\clusternet_roobi_bank ^
  --aois 01 02 03 34 40 41 ^
  --train-frames 102 560 ^
  --val-frames 561 609 ^
  --cluster-quantile 0.90 ^
  --negative-ratio 3
exit /b %errorlevel%
