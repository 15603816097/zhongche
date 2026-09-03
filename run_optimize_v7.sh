#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/val_pred_lgb.npz
  models/val_pred_xgb.npz
  models/val_pred_pca_xgb.npz
  models/ensemble_config.pkl
)

printf '\n[1/4] Required files check...\n'
missing=0
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "请先在完整模型服务器上运行 V6，生成 val_pred_pca_xgb.npz。"
  exit 1
fi

printf '\n[2/4] Syntax check...\n'
python -m py_compile \
  src/trajectory_fusion.py \
  find_best_weight_v7.py \
  official_proxy_evaluator_v7.py

printf '\n[3/4] Search V7 low-rank + peak-preserving candidate...\n'
python find_best_weight_v7.py | tee logs/v7_search.log

printf '\n[4/4] Evaluate V7 candidate...\n'
python official_proxy_evaluator_v7.py | tee logs/proxy_v7.log

printf '\nDone. Important logs:\n'
printf '  logs/v7_search.log\n'
printf '  logs/proxy_v7.log\n'
printf '\nV7 does NOT retrain any model and does NOT overwrite models/ensemble_config.pkl.\n'
printf 'Do NOT submit V7 to the official evaluator until its local gate is reviewed.\n'
