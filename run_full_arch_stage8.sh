#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv-full/bin/python"
OUT_DIR="artifacts/full_arch/freeze"

if [ ! -x "$PY" ]; then
  echo "missing .venv-full; run Stage 1 first" >&2
  exit 2
fi
if [ ! -f full_arch_frozen_gate_v1.json ]; then
  echo "missing tracked frozen gate config" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

dirty="$(git status --porcelain --untracked-files=no)"
if [ -n "$dirty" ]; then
  echo "tracked working tree is dirty; refusing to freeze" >&2
  echo "$dirty" >&2
  exit 2
fi

echo "============================================================"
echo "FULL ARCHITECTURE - STAGE 8 FINAL FREEZE VALIDATION"
echo "============================================================"
echo "This reruns Stage 6 and Stage 7 after switching runtime to"
echo "the tracked frozen gate config. Production V8 is not touched."
echo

bash run_full_arch_stage6.sh 2>&1 | tee "$OUT_DIR/stage6_final.log"
grep -q "STAGE 6 DIRECT RUNTIME: PASS" "$OUT_DIR/stage6_final.log"

echo
bash run_full_arch_stage7.sh 2>&1 | tee "$OUT_DIR/stage7_final.log"
grep -q "STAGE 7 CALLBACK STRESS: PASS" "$OUT_DIR/stage7_final.log"
grep -q "STAGE 7 PASS" "$OUT_DIR/stage7_final.log"

echo
"$PY" full_arch_stage8_freeze.py
