#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check required files...\n'
for f in \
  external_data/corpus/official_finetune_v1.npz \
  models/deep/tcn_v1_pretrain.pt; do
  [ -f "$f" ] || { echo "missing: $f"; exit 1; }
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
python -m py_compile evaluate_tcn_official_loso_calibration.py src/deep/tcn_forecaster.py
python - <<'PY'
import evaluate_tcn_official_loso_calibration
print('TCN official LOSO calibration imports OK')
PY

printf '\n[4/4] Run frozen-TCN leave-one-sequence-out residual calibration...\n'
python evaluate_tcn_official_loso_calibration.py \
  --device "${TCN_DEVICE:-auto}" \
  2>&1 | tee logs/tcn_v1_official_loso_calibration.log

printf '\nDone. Log: logs/tcn_v1_official_loso_calibration.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
