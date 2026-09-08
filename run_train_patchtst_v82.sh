#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models/deep

printf '\n[1/4] Check required files...\n'
for p in \
  external_data/corpus/pretrain_corpus_v1.npz \
  models/deep/patchtst_v1_pretrain.pt; do
  [ -f "$p" ] || { echo "missing $p"; exit 1; }
done

printf '\n[2/4] Check PyTorch/CUDA...\n'
python - <<'PY'
import torch
print('torch version :', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
PY

printf '\n[3/4] Syntax/import check...\n'
python -m py_compile \
  train_patchtst_v82.py \
  src/deep/patchtst_forecaster.py \
  src/deep/patchtst_trend_v82.py

printf '\n[4/4] Fine-tune temperature trend candidate...\n'
START=$(date +%s)
python train_patchtst_v82.py \
  --epochs "${V82_EPOCHS:-35}" \
  --batch-size "${V82_BATCH_SIZE:-64}" \
  --lr "${V82_LR:-0.0002}" \
  --diff-weight "${V82_DIFF_WEIGHT:-0.50}" \
  --direction-weight "${V82_DIRECTION_WEIGHT:-0.12}" \
  --endpoint-weight "${V82_ENDPOINT_WEIGHT:-0.08}" \
  --patience "${V82_PATIENCE:-8}" \
  --device "${V82_DEVICE:-auto}" \
  2>&1 | tee logs/patchtst_v82_train.log
END=$(date +%s)

printf '\nwall elapsed: %d seconds\n' "$((END-START))"
printf 'checkpoint : models/deep/patchtst_v82_trend.pt\n'
printf 'metrics    : models/deep/patchtst_v82_trend_metrics.json\n'
printf 'log        : logs/patchtst_v82_train.log\n'
printf 'NOTE: production V8.1 service/API/callback were not modified.\n'
