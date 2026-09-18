#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv-full/bin/python"
if [ ! -x "$PY" ]; then
  echo "missing .venv-full; run stage 1 first" >&2
  exit 2
fi
export XGB_DEVICE="${XGB_DEVICE:-cuda}"
export LGB_INFER_THREADS="${LGB_INFER_THREADS:-8}"
"$PY" full_arch_stage2_cache.py
