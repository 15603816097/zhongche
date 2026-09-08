#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models

printf '\n[1/4] Check required data/models...\n'
for f in \
  external_data/corpus/official_finetune_v1.npz \
  models/model_lgb.pkl \
  models/scaler.pkl \
  models/model_xgb.pkl \
  models/scaler_xgb.pkl \
  models/ensemble_config.pkl \
  models/model_pca_xgb.pkl \
  models/preprocess_pca_xgb.pkl; do
  [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done

printf '\n[2/4] Syntax/import check...\n'
python -m py_compile v83_final_sprint_diagnostic.py

printf '\n[3/4] Confirm production files are not being edited...\n'
git diff --exit-code -- app.py src/inference.py || {
  echo 'WARNING: app.py or src/inference.py has local changes; aborting final-sprint diagnostic.'
  exit 2
}
python - <<'PY'
import pickle
with open('models/ensemble_config.pkl','rb') as f:
    c=pickle.load(f)
print('version          :', c.get('version'))
print('trajectory_model :', c.get('trajectory_model'))
PY

printf '\n[4/4] Run all-target diagnostic + conservative LOSO search...\n'
START=$(date +%s)
python v83_final_sprint_diagnostic.py 2>&1 | tee logs/v83_final_sprint.log
END=$(date +%s)
printf '\nwall elapsed: %d seconds\n' "$((END-START))"
printf 'metrics : models/v83_final_candidate.json\n'
printf 'log     : logs/v83_final_sprint.log\n'
printf 'NOTE: this runner does not activate V8.3 or modify the production API.\n'
