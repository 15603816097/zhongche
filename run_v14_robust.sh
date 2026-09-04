#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/training_dataset_cache_v2.npz
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
  echo "请在保存完整训练数据缓存的服务器上运行。"
  exit 1
fi

printf '\n[2/3] Syntax check...\n'
python -m py_compile v14_robust_pca_diagnostic.py

printf '\n[3/3] Run V14 robust PCA dual-validation diagnostic...\n'
python v14_robust_pca_diagnostic.py | tee logs/v14_robust.log

printf '\nDone. Important log:\n'
printf '  logs/v14_robust.log\n'
printf '\nV14 is diagnostic only. It does NOT modify models/ensemble_config.pkl, app.py, callback, or online V8.\n'
