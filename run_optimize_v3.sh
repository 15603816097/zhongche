#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required_files=(
  "models/val_pred_lgb.npz"
  "models/val_pred_xgb.npz"
  "models/ensemble_config.pkl"
)

missing=()
for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    missing+=("$f")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf '\n[PRECHECK FAILED] Missing validation/model artifacts:\n'
  for f in "${missing[@]}"; do
    printf '  - %s\n' "$f"
  done
  printf '\nThis optimizer must run on the machine that contains the trained model artifacts.\n'
  printf 'If training was done on the GPU server, SSH to that server and run this script there.\n'
  printf '\nExpected command sequence:\n'
  printf '  cd ~/rail_forecast\n'
  printf '  git pull origin master\n'
  printf '  ls -lh models/val_pred_lgb.npz models/val_pred_xgb.npz models/ensemble_config.pkl\n'
  printf '  bash run_optimize_v3.sh\n\n'
  exit 2
fi

printf '\n[0/4] Artifact precheck... OK\n'
ls -lh "${required_files[@]}"

printf '\n[1/4] Syntax check...\n'
python -m py_compile \
  find_best_weight.py \
  official_proxy_evaluator.py \
  src/inference.py \
  app.py

printf '\n[2/4] Evaluate current ensemble...\n'
python official_proxy_evaluator.py | tee logs/proxy_before_v3.log

printf '\n[3/4] Search trend-aware segmented ensemble V3...\n'
python find_best_weight.py | tee logs/trend_search_v3.log

printf '\n[4/4] Evaluate V3 ensemble...\n'
python official_proxy_evaluator.py | tee logs/proxy_after_v3.log

printf '\nDone. Important logs:\n'
printf '  logs/proxy_before_v3.log\n'
printf '  logs/trend_search_v3.log\n'
printf '  logs/proxy_after_v3.log\n'
printf '\nDo NOT submit to the official evaluator before reviewing these metrics.\n'
