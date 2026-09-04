#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/training_dataset_cache_v2.npz
  models/v14_robust_pca_diagnostic.npz
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
  echo "请先运行 bash run_v14_robust.sh 并保留 V14 artifact。"
  exit 1
fi

printf '\n[2/3] Syntax check...\n'
python -m py_compile v15_blend_v14_diagnostic.py

printf '\n[3/3] Run V15 clean-robust blend diagnostic...\n'
python v15_blend_v14_diagnostic.py | tee logs/v15_blend.log

printf '\nDone. Log:\n'
printf '  logs/v15_blend.log\n'
printf '\nV15 is diagnostic only. It does NOT modify models/ensemble_config.pkl, app.py, callback, or online V8.\n'
