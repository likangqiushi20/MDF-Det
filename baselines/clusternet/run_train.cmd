@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=python"

"%PYTHON%" -B -u -m clusternet.train ^
  --external-repo "%CD%" ^
  --bank generated\clusternet_roobi_bank ^
  --frame-cache generated\frame_cache\cache.json ^
  --truth-cache generated\fair_clusternet_truth.npz ^
  --cluster-checkpoint runs\clusternet\best.pt ^
  --output runs\clusternet ^
  --epochs 12 ^
  --batch-size 256 ^
  --workers 0 ^
  --learning-rate 0.0003 ^
  --jitter 16 ^
  --patience 5 ^
  --train-frame-step 10 ^
  --val-frame-step 5 ^
  --resume
exit /b %errorlevel%
