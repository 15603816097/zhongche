#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/training_dataset_cache_v2.npz
  models/ensemble_config.pkl
  models/val_pred_candidate_v8.npz
  models/val_pred_pca_xgb.npz
  models/preprocess_pca_xgb.pkl
  models/model_pca_xgb.pkl
)

printf '\n[1/3] Required files check...\n'
missing=0
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

printf '\n[2/3] Syntax check...\n'
python -m py_compile \
  src/robust_augmentation.py \
  src/v8_runtime.py \
  train_v15_official.py \
  activate_v15.py \
  rollback_v15.py \
  smoke_test_v15.py

printf '\n[3/3] Train guarded V15 official candidate...\n'
python train_v15_official.py | tee logs/v15_official_prepare.log

printf '\nDone. Log:\n'
printf '  logs/v15_official_prepare.log\n'
printf '\nThis script does NOT activate V15 automatically.\n'
printf 'Only after PASS V15 OFFICIAL PREPARATION should you run: python activate_v15.py\n'
