#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/training_dataset_cache_v2.npz
  models/val_pred_candidate_v8.npz
  models/loso_pca_v12.npz
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
  echo "请先保留 V8 temporal candidate 和 V12 LOSO artifact。"
  exit 1
fi

printf '\n[2/3] Syntax check...\n'
python -m py_compile v13_highfreq_pca_diagnostic.py

printf '\n[3/3] Run V13 dual-validation high-frequency residual diagnostic...\n'
python v13_highfreq_pca_diagnostic.py | tee logs/v13_highfreq.log

printf '\nDone. Important log:\n'
printf '  logs/v13_highfreq.log\n'
printf '\nV13 is diagnostic only. It does NOT modify models/ensemble_config.pkl, app.py, callback, or online V8.\n'
