#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/5] Required files check...\n'
required=(
  models/model_lgb.pkl
  models/scaler.pkl
  models/model_xgb.pkl
  models/scaler_xgb.pkl
  models/val_pred_lgb.npz
  models/val_pred_xgb.npz
  models/ensemble_config.pkl
)
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f"
    exit 1
  fi
done

printf '\n[2/5] Syntax check...\n'
python -m py_compile \
  src/trainer_trend_xgb.py \
  find_best_weight_v5.py \
  official_proxy_evaluator_v5.py \
  src/inference.py \
  app.py

printf '\n[3/5] Train supervised first-difference Trend XGBoost...\n'
python src/trainer_trend_xgb.py | tee logs/train_trend_xgb_v5.log

printf '\n[4/5] Search V5 supervised trend fusion...\n'
python find_best_weight_v5.py | tee logs/trend_search_v5.log

printf '\n[5/5] Evaluate final ensemble...\n'
python official_proxy_evaluator_v5.py | tee logs/proxy_after_v5.log

printf '\nDone. Important logs:\n'
printf '  logs/train_trend_xgb_v5.log\n'
printf '  logs/trend_search_v5.log\n'
printf '  logs/proxy_after_v5.log\n'
printf '\nDo NOT submit to the official evaluator until the V5 search conclusion is reviewed.\n'
