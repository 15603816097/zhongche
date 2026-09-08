#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models

printf '\n[1/3] Check source V8.3 candidate...\n'
[ -f models/v83_final_candidate.json ] || {
  echo 'missing models/v83_final_candidate.json; run bash run_v83_final_sprint.sh first'
  exit 1
}

printf '\n[2/3] Syntax/import check...\n'
python -m py_compile v83_safe_shrink_diagnostic.py

printf '\n[3/3] Run final 5/5 safety shrink ablation...\n'
python v83_safe_shrink_diagnostic.py 2>&1 | tee logs/v83_safe_shrink.log

printf '\nmetrics : models/v83_final_safe_candidate.json\n'
printf 'log     : logs/v83_safe_shrink.log\n'
printf 'NOTE: production service/API/callback are still untouched.\n'
