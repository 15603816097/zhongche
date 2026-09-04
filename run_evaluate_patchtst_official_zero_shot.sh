#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check required files...\n'
for f in \
  external_data/corpus/official_finetune_v1.npz \
  models/deep/patchtst_v1_pretrain.pt
do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
printf 'required files OK\n'

printf '\n[2/4] Check PyTorch/CUDA...\n'
python - <<'PY'
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
PY

printf '\n[3/4] Syntax/import check...\n'
python -m py_compile evaluate_patchtst_official_zero_shot.py src/deep/patchtst_forecaster.py
python - <<'PY'
from evaluate_patchtst_official_zero_shot import main
print('PatchTST official zero-shot imports OK')
PY

printf '\n[4/4] Evaluate pretrained PatchTST on official five-sequence corpus WITHOUT fine-tuning...\n'
python evaluate_patchtst_official_zero_shot.py \
  --device "${PATCHTST_DEVICE:-auto}" \
  2>&1 | tee logs/patchtst_v1_official_zero_shot.log

printf '\nDone. Log: logs/patchtst_v1_official_zero_shot.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
