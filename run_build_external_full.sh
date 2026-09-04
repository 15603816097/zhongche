#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/4] Check data dependencies...\n'
python - <<'PY'
import importlib.util
mods = ['numpy', 'pandas', 'scipy', 'nptdms']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing packages: ' + ', '.join(missing))
print('dependencies OK')
PY

printf '\n[2/4] Syntax/import check...\n'
python -m py_compile build_external_dataset.py \
  src/external/kaist_adapter.py \
  src/external/metropt_adapter.py \
  src/external/signal_processing.py \
  src/external/domain_calibrator.py
python - <<'PY'
from src.external import KaistAdapter, MetroPTAdapter, OfficialDomainCalibrator
print('external adapter imports OK')
PY

printf '\n[3/4] Build FULL KAIST + FULL MetroPT source-domain corpus...\n'
START=$(date +%s)
python build_external_dataset.py --feature-hz 10.0 | tee logs/external_build_full.log
END=$(date +%s)
printf '\nfull build elapsed: %d seconds\n' "$((END-START))"

printf '\n[4/4] Validate generated corpus...\n'
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path('.')
processed = root / 'external_data' / 'processed'
manifest_path = processed / 'manifest.json'
if not manifest_path.is_file():
    raise SystemExit('manifest missing')
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
kaist = manifest.get('kaist', [])
metro = manifest.get('metropt', {})

print('KAIST processed conditions:', len(kaist))
print('KAIST with acoustic      :', sum(bool(x.get('has_acoustic')) for x in kaist))
print('MetroPT rows             :', metro.get('rows'))
print('MetroPT segments         :', metro.get('contiguous_segments'))

if len(kaist) < 40:
    raise SystemExit(f'expected >=40 KAIST conditions, got {len(kaist)}')
if int(metro.get('rows') or 0) < 100000:
    raise SystemExit('MetroPT full build unexpectedly small')

run_files = sorted((processed / 'kaist_runs').glob('*.csv'))
print('KAIST CSV files          :', len(run_files))
if len(run_files) < len(kaist):
    raise SystemExit('some KAIST outputs are missing')

# Check representative files for required source-domain columns and finite values.
required = {'time_s', 'vibration_rms', 'temperature_c', 'current_a', 'acoustic_db'}
for path in run_files[:3]:
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'{path}: missing columns {sorted(missing)}')
    for col in ['vibration_rms', 'temperature_c', 'current_a']:
        if not pd.to_numeric(df[col], errors='coerce').notna().any():
            raise SystemExit(f'{path}: {col} has no finite values')

print('FULL EXTERNAL BUILD VALIDATION: PASS')
PY

printf '\nDone. Log: logs/external_build_full.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
