#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

printf '\n[1/4] Check required files...\n'
python - <<'PY'
from pathlib import Path

required = [
    Path('external_data/corpus/official_finetune_v1.npz'),
    Path('models/deep/patchtst_v1_pretrain.pt'),
    Path('models/model_lgb.pkl'),
    Path('models/scaler.pkl'),
    Path('models/model_xgb.pkl'),
    Path('models/scaler_xgb.pkl'),
    Path('models/ensemble_config.pkl'),
    Path('models/model_pca_xgb.pkl'),
    Path('models/preprocess_pca_xgb.pkl'),
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    print('missing files:')
    for p in missing:
        print('  ', p)
    raise SystemExit(2)
print('required files OK')
PY

printf '\n[2/4] Check PyTorch/CUDA...\n'
python - <<'PY'
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
PY

printf '\n[3/4] Syntax/import check...\n'
python -m py_compile evaluate_patchtst_temperature_candidate.py
python - <<'PY'
import evaluate_patchtst_temperature_candidate as m
print('imports OK')
print('target      :', m.TEMP_NAME)
print('weight grid :', m.WEIGHT_GRID.tolist())
PY

printf '\n[4/4] Evaluate conservative temperature-only PatchTST candidate vs current V8...\n'
python evaluate_patchtst_temperature_candidate.py 2>&1 | tee logs/patchtst_temperature_candidate.log

printf '\nDone. Log: logs/patchtst_temperature_candidate.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
