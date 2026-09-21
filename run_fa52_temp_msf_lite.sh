#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-/root/zhongche_v8speed}"
PY="${PY:-$SOURCE_ROOT/.venv-v8/bin/python}"

echo "======================================================================"
echo "FA52.895 SAFE EXPERIMENT - TEMPERATURE MSF-LITE"
echo "======================================================================"
echo "workspace      : $ROOT"
echo "source runtime : $SOURCE_ROOT"
echo "python         : $PY"
echo "production     : untouched"
echo

if [ ! -x "$PY" ]; then
  echo "missing Python runtime: $PY" >&2
  exit 2
fi

# Models and official five sequences are intentionally reused read-only from
# the validated production workspace. The experimental source tree stays
# isolated from production.
if [ ! -e "$ROOT/models" ]; then
  ln -s "$SOURCE_ROOT/models" "$ROOT/models"
fi
if [ ! -e "$ROOT/data" ]; then
  ln -s "$SOURCE_ROOT/data" "$ROOT/data"
fi

if [ ! -f "$ROOT/full_arch_frozen_gate_v1.json" ]; then
  echo "missing full_arch_frozen_gate_v1.json; wrong branch?" >&2
  exit 2
fi

"$PY" - <<'PY'
import json
from pathlib import Path
p=Path("full_arch_frozen_gate_v1.json")
d=json.loads(p.read_text(encoding="utf-8"))
assert d.get("global_gate_pass") is True
assert d.get("enabled_targets") == ["speed_rpm","acoustic_db","pressure_kpa"]
print("frozen Full Architecture config: PASS")
PY

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
LGB_INFER_THREADS=2 \
"$PY" fa52_temp_msf_lite_diagnostic.py
