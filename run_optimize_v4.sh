#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/5] Required files check...\n'
required=(
  models/val_pred_lgb.npz
  models/val_pred_xgb.npz
  models/ensemble_config.pkl
)
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f"
    exit 1
  fi
done

printf '\n[2/5] Syntax check...\n'
python -m py_compile \
  src/trend_pattern.py \
  find_best_weight_v4.py \
  official_proxy_evaluator_v4.py \
  src/inference.py \
  app.py

printf '\n[3/5] Evaluate V3 current baseline...\n'
python official_proxy_evaluator_v4.py | tee logs/proxy_before_v4.log

printf '\n[4/5] Search V4 adaptive pattern trend ensemble...\n'
python find_best_weight_v4.py | tee logs/trend_search_v4.log

printf '\n[5/5] Evaluate final ensemble...\n'
python official_proxy_evaluator_v4.py | tee logs/proxy_after_v4.log

printf '\nDone. Important logs:\n'
printf '  logs/proxy_before_v4.log\n'
printf '  logs/trend_search_v4.log\n'
printf '  logs/proxy_after_v4.log\n'
printf '\nDo NOT submit to the official evaluator until the V4 search conclusion is reviewed.\n'
