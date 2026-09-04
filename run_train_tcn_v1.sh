#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check corpus...\n'
[ -f external_data/corpus/pretrain_corpus_v1.npz ] || {
  echo 'missing external_data/corpus/pretrain_corpus_v1.npz';
  echo 'run: bash run_build_pretrain_corpus.sh';
  exit 1;
}

printf '\n[2/4] Check PyTorch...\n'
python - <<'PY'
try:
    import torch
except Exception as exc:
    raise SystemExit('PyTorch unavailable: ' + repr(exc))
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
else:
    print('NOTE: training will run on CPU. For the formal run, prefer the RTX 4090 server.')
PY

printf '\n[3/4] Syntax/import check...\n'
python -m py_compile train_tcn_v1.py src/deep/tcn_forecaster.py
python - <<'PY'
from src.deep.tcn_forecaster import MaskedTCNForecaster, TCNConfig
m = MaskedTCNForecaster(TCNConfig(hidden_channels=16, num_blocks=2))
print('TCN import OK; params:', sum(p.numel() for p in m.parameters()))
PY

printf '\n[4/4] Train/evaluate TCN V1...\n'
START=$(date +%s)
python train_tcn_v1.py \
  --epochs "${TCN_EPOCHS:-80}" \
  --batch-size "${TCN_BATCH_SIZE:-64}" \
  --lr "${TCN_LR:-0.001}" \
  --patience "${TCN_PATIENCE:-12}" \
  --device "${TCN_DEVICE:-auto}" \
  2>&1 | tee logs/tcn_v1_train.log
END=$(date +%s)
printf '\nwall elapsed: %d seconds\n' "$((END-START))"
printf 'Done. Log: logs/tcn_v1_train.log\n'
printf 'NOTE: online V8/API/callback files were not modified.\n'
