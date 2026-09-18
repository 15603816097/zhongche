#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv-full/bin/python"
if [ ! -x "$PY" ]; then
  echo "missing .venv-full; run stage1 first" >&2
  exit 2
fi
if [ ! -f artifacts/full_arch/expert_cache/manifest.json ]; then
  echo "missing Stage 2 cache; run Stage 2 first" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
"$PY" full_arch_stage3_deep_loso.py \
  --input-length 144 \
  --stride 2 \
  --epochs 30 \
  --final-epochs 35 \
  --batch-size 64
