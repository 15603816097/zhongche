#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/val_pred_lgb.npz
  models/val_pred_xgb.npz
  models/ensemble_config.pkl
  models/val_pred_pca_xgb.npz
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
  echo "请在保存完整 models 训练产物的服务器上运行。"
  exit 1
fi

printf '\n[2/4] Syntax check...\n'
python -m py_compile \
  find_best_weight_v8.py \
  official_proxy_evaluator_v8.py \
  src/trajectory_fusion.py

printf '\n[3/4] Search V8 source-aware low-rank + peak-preserving candidate...\n'
python find_best_weight_v8.py | tee logs/v8_search.log

printf '\n[4/4] Evaluate V8 candidate...\n'
python official_proxy_evaluator_v8.py | tee logs/proxy_v8.log

printf '\nDone. Important logs:\n'
printf '  logs/v8_search.log\n'
printf '  logs/proxy_v8.log\n'
printf '\nV8 does NOT retrain any model and does NOT overwrite models/ensemble_config.pkl.\n'
printf 'Do NOT submit V8 until the local gate is reviewed.\n'
