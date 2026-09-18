#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv-full/bin/python"
if [ ! -x "$PY" ]; then
  echo "missing .venv-full; run Stage 1 first" >&2
  exit 2
fi
if [ ! -f artifacts/full_arch/expert_cache/manifest.json ]; then
  echo "missing Stage 2 cache; run Stage 2 first" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
"$PY" full_arch_stage4_confidence_ood.py
