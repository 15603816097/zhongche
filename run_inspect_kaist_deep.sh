#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/4] Check KAIST files...\n'
[ -d external_data/kaist ] || { echo 'missing external_data/kaist'; exit 1; }

printf '\n[2/4] Check data dependencies...\n'
python - <<'PY'
import importlib.util
mods = ['numpy', 'scipy', 'nptdms']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print('missing:', ', '.join(missing))
    print('install with: python -m pip install -r requirements_data.txt')
    raise SystemExit(2)
print('dependencies OK')
PY

printf '\n[3/4] Syntax check...\n'
python -m py_compile inspect_kaist_deep.py

printf '\n[4/4] Deep inspect KAIST MAT/TDMS channels...\n'
python inspect_kaist_deep.py | tee logs/kaist_deep_inspection.log

printf '\nDone. Log: logs/kaist_deep_inspection.log\n'
