#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check corpus...\n'
[ -f external_data/corpus/pretrain_corpus_v1.npz ] || {
  echo 'missing external_data/corpus/pretrain_corpus_v1.npz'
  exit 1
}

printf '\n[2/4] Check PyTorch/CUDA...\n'
python - <<'PY'
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
PY

printf '\n[3/4] Syntax/import check...\n'
python -m py_compile train_patchtst_v1.py src/deep/patchtst_forecaster.py
python - <<'PY'
from src.deep.patchtst_forecaster import MaskedPatchTSTForecaster, PatchTSTConfig
m = MaskedPatchTSTForecaster(PatchTSTConfig(d_model=32, n_heads=4, num_layers=1, dim_feedforward=64))
print('PatchTST import OK; params:', sum(p.numel() for p in m.parameters()))
PY

printf '\n[4/4] Train/evaluate PatchTST V1...\n'
START=$(date +%s)
python train_patchtst_v1.py \
  --epochs "${PATCHTST_EPOCHS:-60}" \
  --batch-size "${PATCHTST_BATCH_SIZE:-64}" \
  --lr "${PATCHTST_LR:-0.0008}" \
  --patience "${PATCHTST_PATIENCE:-10}" \
  --device "${PATCHTST_DEVICE:-auto}" \
  2>&1 | tee logs/patchtst_v1_train.log
END=$(date +%s)
printf '\nwall elapsed: %d seconds\n' "$((END-START))"
printf 'Done. Log: logs/patchtst_v1_train.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
