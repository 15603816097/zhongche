#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/val_pred_lgb.npz
  models/val_pred_xgb.npz
  models/ensemble_config.pkl
  models/training_dataset_cache_v2.npz
)

printf '\n[1/5] Required files check...\n'
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

printf '\n[2/5] Syntax check...\n'
python -m py_compile \
  src/trainer_pca_xgb.py \
  find_best_weight_v6.py \
  official_proxy_evaluator_v6.py

printf '\n[3/5] Train low-rank PCA trajectory XGBoost...\n'
python src/trainer_pca_xgb.py | tee logs/train_pca_xgb_v6.log

printf '\n[4/5] Search safe V6 PCA blend...\n'
python find_best_weight_v6.py | tee logs/pca_search_v6.log

printf '\n[5/5] Evaluate V6 candidate...\n'
python official_proxy_evaluator_v6.py | tee logs/proxy_v6.log

printf '\nDone. Important logs:\n'
printf '  logs/train_pca_xgb_v6.log\n'
printf '  logs/pca_search_v6.log\n'
printf '  logs/proxy_v6.log\n'
printf '\nV6 never overwrites models/ensemble_config.pkl.\n'
printf 'Do NOT submit V6 candidate to the official evaluator until its local gate is reviewed.\n'
