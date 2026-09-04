#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

printf '\n[1/3] Check external_data...\n'
[ -d external_data/kaist ] || { echo 'missing external_data/kaist'; exit 1; }
[ -f 'external_data/metropt/MetroPT3(AirCompressor).csv' ] || { echo 'missing MetroPT3(AirCompressor).csv'; exit 1; }

printf '\n[2/3] Check Python dependencies...\n'
python - <<'PY'
import importlib.util
mods = ['pandas', 'scipy']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing required packages: ' + ', '.join(missing))
print('required packages OK')
print('nptdms:', 'installed' if importlib.util.find_spec('nptdms') else 'missing (TDMS channel details will be skipped)')
PY

printf '\n[3/3] Inspect actual KAIST + MetroPT files...\n'
python inspect_external_data.py | tee logs/external_data_inspection.log

printf '\nDone. Log: logs/external_data_inspection.log\n'
