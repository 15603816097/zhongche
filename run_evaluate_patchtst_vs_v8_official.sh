#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check PatchTST/offline corpus...\n'
for f in \
  external_data/corpus/official_finetune_v1.npz \
  models/deep/patchtst_v1_pretrain.pt; do
  [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
printf 'PatchTST/corpus OK\n'

printf '\n[2/4] Check current V8 model pack...\n'
MISSING=0
for f in \
  models/model_lgb.pkl \
  models/scaler.pkl \
  models/model_xgb.pkl \
  models/scaler_xgb.pkl \
  models/ensemble_config.pkl \
  models/model_pca_xgb.pkl \
  models/preprocess_pca_xgb.pkl; do
  if [ ! -f "$f" ]; then
    echo "missing: $f"
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo
  echo 'V8 MODEL PACK IS INCOMPLETE.'
  echo 'Copy the seven files above from the machine/server where V8 52.73 is stored.'
  echo 'Do NOT retrain or regenerate them for this comparison.'
  exit 2
fi
printf 'V8 model pack OK\n'

printf '\n[3/4] Syntax/import check...\n'
python -m py_compile evaluate_patchtst_vs_v8_official.py
python - <<'PY'
from evaluate_patchtst_vs_v8_official import ACTIVE_TARGETS, WEIGHT_GRID
print('imports OK')
print('active targets:', ACTIVE_TARGETS)
print('weight grid   :', WEIGHT_GRID.tolist())
PY

printf '\n[4/4] Compare frozen PatchTST with current V8 using sequence-level LOSO blend...\n'
python evaluate_patchtst_vs_v8_official.py 2>&1 | tee logs/patchtst_vs_v8_official_loso.log

printf '\nDone. Log: logs/patchtst_vs_v8_official_loso.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
