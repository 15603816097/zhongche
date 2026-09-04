#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs external_data/processed

printf '\n[1/4] Check dependencies...\n'
python - <<'PY'
import importlib.util
mods = ['numpy', 'pandas', 'scipy', 'nptdms']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing: ' + ', '.join(missing) + '\nrun: python -m pip install -r requirements_data.txt')
print('dependencies OK')
PY

printf '\n[2/4] Syntax/import check...\n'
python -m py_compile \
  src/external/signal_processing.py \
  src/external/kaist_adapter.py \
  src/external/metropt_adapter.py \
  src/external/domain_calibrator.py \
  build_external_dataset.py
python - <<'PY'
from src.external import KaistAdapter, MetroPTAdapter, OfficialDomainCalibrator
print('external adapter imports OK')
PY

printf '\n[3/4] Build a small real-data smoke corpus...\n'
python build_external_dataset.py \
  --feature-hz 10 \
  --kaist-limit 2 \
  --metro-rows 20000 | tee logs/external_build_smoke.log

printf '\n[4/4] Show generated files...\n'
find external_data/processed -maxdepth 2 -type f -printf '%p  %k KB\n' | sort

printf '\nDone. Log: logs/external_build_smoke.log\n'
