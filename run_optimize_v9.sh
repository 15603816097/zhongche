#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/training_dataset_cache_v2.npz
  models/val_pred_lgb.npz
  models/val_pred_xgb.npz
  models/val_pred_candidate_v8.npz
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
  echo "请在保存完整 V8 / validation 产物的服务器上运行。"
  exit 1
fi

printf '\n[2/4] Syntax check...\n'
python -m py_compile \
  src/template_shape.py \
  find_best_weight_v9.py \
  official_proxy_evaluator_v9.py

printf '\n[3/4] Search V9 causal sequence-template shape fusion...\n'
python find_best_weight_v9.py | tee logs/v9_search.log

printf '\n[4/4] Evaluate V9 candidate...\n'
python official_proxy_evaluator_v9.py | tee logs/proxy_v9.log

printf '\nDone. Important logs:\n'
printf '  logs/v9_search.log\n'
printf '  logs/proxy_v9.log\n'
printf '\nV9 is OFFLINE ONLY and never overwrites models/ensemble_config.pkl.\n'
printf 'Do NOT restart API or submit V9 until its local result is reviewed.\n'
