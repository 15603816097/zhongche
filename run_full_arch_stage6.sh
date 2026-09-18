#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv-full/bin/python"
if [ ! -x "$PY" ]; then
  echo "missing .venv-full; run Stage 1 first" >&2
  exit 2
fi
if [ ! -f models/full_arch/dynamic_gate_config.json ]; then
  echo "missing Stage 5 gate config; run Stage 5 first" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export LGB_INFER_THREADS="${LGB_INFER_THREADS:-8}"
"$PY" test_full_arch_stage6_runtime.py
