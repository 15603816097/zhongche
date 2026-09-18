#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv-full/bin/python"
if [ ! -x "$PY" ]; then
  echo "missing .venv-full; run Stage 1 first" >&2
  exit 2
fi
if [ ! -f artifacts/full_arch/confidence_ood/analog_confidence_ood.npz ]; then
  echo "missing Stage 4 artifacts; run Stage 4 first" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
"$PY" full_arch_stage5_dynamic_gate.py
