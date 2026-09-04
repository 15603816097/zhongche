#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

required=(
  models/training_dataset_cache_v2.npz
  models/v15_blend_v14_diagnostic.npz
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
  echo "请先完成 V15，再运行 V16。"
  exit 1
fi

printf '\n[2/3] Syntax check...\n'
python -m py_compile v16_stress_blend_diagnostic.py

printf '\n[3/3] Run V16 final stress validation...\n'
python v16_stress_blend_diagnostic.py | tee logs/v16_stress.log

printf '\nDone. Important log:\n'
printf '  logs/v16_stress.log\n'
printf '\nV16 is diagnostic only. It does NOT modify models/ensemble_config.pkl, app.py, callback, or online V8.\n'
