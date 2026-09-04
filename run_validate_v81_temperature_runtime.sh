#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

echo
echo "[1/4] Check required candidate files..."
python - <<'PY'
from pathlib import Path
required = [
    Path("models/deep/patchtst_v1_pretrain.pt"),
    Path("external_data/corpus/official_finetune_v1.npz"),
    Path("models/model_lgb.pkl"),
    Path("models/scaler.pkl"),
    Path("models/model_xgb.pkl"),
    Path("models/scaler_xgb.pkl"),
    Path("models/ensemble_config.pkl"),
    Path("models/model_pca_xgb.pkl"),
    Path("models/preprocess_pca_xgb.pkl"),
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    print("missing:")
    for p in missing:
        print(" ", p)
    raise SystemExit(2)
print("required files OK")
PY

echo
echo "[2/4] Check PyTorch/CUDA..."
python - <<'PY'
import torch
print("torch version :", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu           :", torch.cuda.get_device_name(0))
PY

echo
echo "[3/4] Syntax/import check..."
python -m py_compile \
  src/deep/patchtst_temperature_runtime.py \
  validate_v81_temperature_runtime.py
python - <<'PY'
from src.deep.patchtst_temperature_runtime import PATCHTST_TEMPERATURE_WEIGHT
print("runtime imports OK")
print("temperature weight:", PATCHTST_TEMPERATURE_WEIGHT)
PY

echo
echo "[4/4] Validate exact runtime parity, V8 isolation and latency..."
python validate_v81_temperature_runtime.py \
  --repeats "${V81_BENCH_REPEATS:-30}" \
  2>&1 | tee logs/v81_temperature_runtime_validation.log

echo
echo "Done. Log: logs/v81_temperature_runtime_validation.log"
echo "NOTE: online V8/API/callback/config were not modified."
