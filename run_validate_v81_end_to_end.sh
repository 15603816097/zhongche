#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

export V81_TEMPERATURE_ENABLED="${V81_TEMPERATURE_ENABLED:-1}"
export V81_TEMPERATURE_WEIGHT="${V81_TEMPERATURE_WEIGHT:-0.15}"
export V81_PATCHTST_DEVICE="${V81_PATCHTST_DEVICE:-auto}"
export V81_STRICT_CANDIDATE="1"

echo
echo "[1/4] Check required files..."
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
    for p in missing:
        print('missing:', p)
    raise SystemExit(2)
print('required files OK')
PY

echo
echo "[2/4] Check PyTorch/CUDA..."
python - <<'PY'
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
PY

echo
echo "[3/4] Syntax/import check..."
python -m py_compile \
  src/inference_v81.py \
  validate_v81_end_to_end.py
python - <<'PY'
from src.inference_v81 import (
    V81_PATCHTST_DEVICE,
    V81_TEMPERATURE_ENABLED,
    V81_TEMPERATURE_WEIGHT,
)
print('V8.1 wrapper imports OK')
print('enabled :', V81_TEMPERATURE_ENABLED)
print('weight  :', V81_TEMPERATURE_WEIGHT)
print('device  :', V81_PATCHTST_DEVICE)
PY

echo
echo "[4/4] Run end-to-end candidate validation..."
python validate_v81_end_to_end.py 2>&1 | tee logs/v81_end_to_end_validation.log

echo
echo "Done. Log: logs/v81_end_to_end_validation.log"
echo "NOTE: app.py / callback / ensemble_config.pkl were not modified."
