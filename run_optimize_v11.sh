#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/3] Required files check...\n'
required=(
  models/training_dataset_cache_v2.npz
  models/model_lgb.pkl
  models/model_xgb.pkl
  models/model_pca_xgb.pkl
  models/preprocess_pca_xgb.pkl
  models/ensemble_config.pkl
)
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
  src/legacy_shape_teacher.py \
  find_best_weight_v11.py

printf '\n[3/3] Run V11 legacy future-shape boundary/stress diagnostic...\n'
python find_best_weight_v11.py | tee logs/v11_diagnostic.log

printf '\nDone. Log:\n'
printf '  logs/v11_diagnostic.log\n'
printf '\nV11 is diagnostic/offline only. It never overwrites models/ensemble_config.pkl.\n'
