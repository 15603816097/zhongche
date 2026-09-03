#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/4] Syntax check...\n'
python -m py_compile \
  find_best_weight.py \
  official_proxy_evaluator.py \
  src/inference.py \
  app.py

printf '\n[2/4] Evaluate current ensemble...\n'
python official_proxy_evaluator.py | tee logs/proxy_before_v3.log

printf '\n[3/4] Search trend-aware segmented ensemble V3...\n'
python find_best_weight.py | tee logs/trend_search_v3.log

printf '\n[4/4] Evaluate V3 ensemble...\n'
python official_proxy_evaluator.py | tee logs/proxy_after_v3.log

printf '\nDone. Important logs:\n'
printf '  logs/proxy_before_v3.log\n'
printf '  logs/trend_search_v3.log\n'
printf '  logs/proxy_after_v3.log\n'
printf '\nDo NOT submit to the official evaluator before reviewing these metrics.\n'
