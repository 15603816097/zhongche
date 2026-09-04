#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check required files...\n'
[ -f external_data/corpus/official_finetune_v1.npz ] || {
  echo 'missing external_data/corpus/official_finetune_v1.npz';
  exit 1;
}
[ -f models/deep/tcn_v1_pretrain.pt ] || {
  echo 'missing models/deep/tcn_v1_pretrain.pt';
  echo 'run: bash run_train_tcn_v1.sh';
  exit 1;
}

echo 'required files OK'

printf '\n[2/4] Check PyTorch/CUDA...\n'
python - <<'PY'
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
PY

printf '\n[3/4] Syntax/import check...\n'
python -m py_compile evaluate_tcn_official_zero_shot.py src/deep/tcn_forecaster.py
python - <<'PY'
from src.deep.tcn_forecaster import MaskedTCNForecaster, TCNConfig
print('TCN official zero-shot imports OK')
PY

printf '\n[4/4] Evaluate pretrained TCN on official five-sequence corpus WITHOUT fine-tuning...\n'
python evaluate_tcn_official_zero_shot.py \
  --device "${TCN_DEVICE:-auto}" \
  2>&1 | tee logs/tcn_v1_official_zero_shot.log

printf '\nDone. Log: logs/tcn_v1_official_zero_shot.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
