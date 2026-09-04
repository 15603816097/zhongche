#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo
echo "[1/4] Check required candidate files..."
required=(
  "app.py"
  "app_v81.py"
  "src/inference_v81.py"
  "src/deep/patchtst_temperature_runtime.py"
  "validate_v81_api_candidate.py"
  "models/deep/patchtst_v1_pretrain.pt"
  "external_data/corpus/official_finetune_v1.npz"
)
for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "missing: $f"
    exit 2
  fi
done
echo "required files OK"

echo
echo "[2/4] Confirm verified base app has no working-tree edits..."
if ! git diff --exit-code -- app.py >/dev/null; then
  echo "ERROR: app.py has local edits. Refusing candidate smoke."
  git diff -- app.py
  exit 2
fi
echo "app.py working tree unchanged"

echo
echo "[3/4] Syntax/import check..."
python -m py_compile \
  app_v81.py \
  src/inference_v81.py \
  src/deep/patchtst_temperature_runtime.py \
  validate_v81_api_candidate.py
python - <<'PY'
import app_v81
from src.inference_v81 import V81_TEMPERATURE_ENABLED, V81_TEMPERATURE_WEIGHT
print("candidate app import OK")
print("enabled :", V81_TEMPERATURE_ENABLED)
print("weight  :", V81_TEMPERATURE_WEIGHT)
print("version :", app_v81.APP_VERSION)
PY

echo
echo "[4/4] Run V8.1 candidate API integration smoke..."
mkdir -p logs
python validate_v81_api_candidate.py 2>&1 | tee logs/v81_api_candidate_validation.log

echo
echo "Done. Log: logs/v81_api_candidate_validation.log"
echo "NOTE: app.py / verified callback implementation / ensemble_config.pkl were not modified."
