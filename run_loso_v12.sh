#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/3] Required files check...\n'
required=(
  models/training_dataset_cache_v2.npz
)
missing=0
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "请在训练数据和模型完整的服务器上运行。"
  exit 1
fi

printf '\n[2/3] Syntax check...\n'
python -m py_compile loso_pca_diagnostic_v12.py

printf '\n[3/3] Run V12 leave-one-sequence-out PCA diagnostic...\n'
python loso_pca_diagnostic_v12.py | tee logs/loso_v12.log

printf '\nDone. Important log:\n'
printf '  logs/loso_v12.log\n'
printf '\nV12 is diagnostic only. It does NOT modify models/ensemble_config.pkl, app.py, callback, or online V8.\n'
