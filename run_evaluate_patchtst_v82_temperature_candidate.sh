#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check required files...\n'
for p in \
  external_data/corpus/official_finetune_v1.npz \
  models/deep/patchtst_v1_pretrain.pt \
  models/deep/patchtst_v82_trend.pt \
  models/model_lgb.pkl \
  models/scaler.pkl \
  models/model_xgb.pkl \
  models/scaler_xgb.pkl \
  models/ensemble_config.pkl \
  models/model_pca_xgb.pkl \
  models/preprocess_pca_xgb.pkl; do
  [ -f "$p" ] || { echo "missing $p"; exit 1; }
done

printf '\n[2/4] Syntax/import check...\n'
python -m py_compile \
  evaluate_patchtst_v82_temperature_candidate.py \
  src/deep/patchtst_forecaster.py \
  src/inference.py

printf '\n[3/4] Confirm current V8 config...\n'
python - <<'PY'
import pickle
from pathlib import Path
p = Path('models/ensemble_config.pkl')
with p.open('rb') as f:
    cfg = pickle.load(f)
print('version          :', cfg.get('version'))
print('trajectory_model :', cfg.get('trajectory_model'))
PY

printf '\n[4/4] Run five-sequence LOSO: V8 vs V8.1 vs V8.2...\n'
python evaluate_patchtst_v82_temperature_candidate.py \
  2>&1 | tee logs/patchtst_v82_official_loso.log

printf '\nmetrics : models/deep/patchtst_v82_temperature_candidate.json\n'
printf 'log     : logs/patchtst_v82_official_loso.log\n'
printf 'IMPORTANT: only a PASS is eligible for later runtime/API integration.\n'
printf 'Production app.py/callback/ensemble_config.pkl were not modified.\n'
