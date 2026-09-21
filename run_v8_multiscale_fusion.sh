#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="$ROOT/.venv-v8/bin/python"

if [ ! -x "$PY" ]; then
  echo "missing $PY; deploy V8 baseline first" >&2
  exit 2
fi

echo "======================================================================"
echo "V8 MULTI-SCALE FUSION OFFLINE SEARCH"
echo "======================================================================"
echo "production 8800 remains untouched"
echo

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
LGB_ESTIMATOR_THREADS=1 \
XGB_INFER_THREADS=1 \
PCA_XGB_INFER_THREADS=1 \
LGB_INFER_THREADS=2 \
"$PY" v8_multiscale_fusion_diagnostic.py
