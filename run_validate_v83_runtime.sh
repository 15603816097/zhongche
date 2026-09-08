#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models

printf '\n[1/5] Check V8.3 safe candidate output...\n'
[ -f models/v83_final_safe_candidate.json ] || {
  echo 'missing models/v83_final_safe_candidate.json; run bash run_v83_safe_shrink.sh first'
  exit 1
}

printf '\n[2/5] Confirm safe candidate PASS...\n'
python - <<'PY'
import json
from pathlib import Path
p = Path('models/v83_final_safe_candidate.json')
d = json.loads(p.read_text(encoding='utf-8'))
print('offline_gate_pass :', d.get('offline_gate_pass'))
print('enabled_targets   :', d.get('enabled_targets'))
print('global_rmse_ratio :', d.get('global_rmse_ratio'))
print('proxy_gain_pct    :', d.get('global_proxy_gain_pct'))
for name in d.get('enabled_targets', []):
    item = d['targets'][name]
    print(f"  {name:16s} runtime_scale={item.get('runtime_scale')}")
if not d.get('offline_gate_pass'):
    raise SystemExit('safe candidate is not PASS')
PY

printf '\n[3/5] Syntax/import check...\n'
python -m py_compile src/inference_v83.py app_v83.py validate_v83_runtime.py

printf '\n[4/5] Verify accepted API/callback file is untouched...\n'
git diff --exit-code -- app.py
printf 'app.py git diff: CLEAN\n'

printf '\n[5/5] Run strict runtime parity gate...\n'
python validate_v83_runtime.py 2>&1 | tee logs/v83_runtime_validation.log

printf '\nDone. Log: logs/v83_runtime_validation.log\n'
printf 'IMPORTANT: production service was NOT restarted by this runner.\n'
