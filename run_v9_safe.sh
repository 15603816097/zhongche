#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEPLOY_MODELS="${V8_DEPLOY_MODELS:-/root/rail_forecast_v8_deploy/models}"

REQUIRED=(
  model_lgb.pkl
  scaler.pkl
  model_xgb.pkl
  scaler_xgb.pkl
  ensemble_config.pkl
  model_pca_xgb.pkl
  preprocess_pca_xgb.pkl
)

echo "[1/5] Check official data..."
for s in sequence0001 sequence0002 sequence0003 sequence0004 sequence0005; do
  test -f "data/raw/${s}/history.csv"
  test -f "data/raw/${s}/future.csv"
done
echo "official data OK"

echo
echo "[2/5] Link exact V8 model pack without copying 2GB..."
mkdir -p models
for f in "${REQUIRED[@]}"; do
  if [ ! -e "models/$f" ]; then
    test -f "$DEPLOY_MODELS/$f" || {
      echo "missing model: $DEPLOY_MODELS/$f" >&2
      exit 1
    }
    ln -s "$DEPLOY_MODELS/$f" "models/$f"
  fi
done
ls -lh models/model_lgb.pkl models/model_xgb.pkl models/model_pca_xgb.pkl models/ensemble_config.pkl

echo
echo "[3/5] Verify production V8 on port 8800 is healthy and untouched..."
python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen('http://127.0.0.1:8800/health', timeout=5) as r:
    body = json.load(r)
print('production health:', body)
if body.get('ensemble_version') != 8 or not body.get('model_loaded'):
    raise SystemExit('production is not healthy V8')
PY

echo
echo "[4/5] Syntax + V8 config check..."
python -m py_compile v9_safe_diagnostic.py v9_analog_multiscale_diagnostic.py
python - <<'PY'
import pickle
from pathlib import Path

p = Path('models/ensemble_config.pkl')
with p.open('rb') as f:
    cfg = pickle.load(f)
print('ensemble version :', cfg.get('version'))
print('trajectory model :', cfg.get('trajectory_model'))
if int(cfg.get('version', -1)) != 8:
    raise SystemExit('not V8')
PY

echo
echo "[5/5] Run V9-Safe diagnostic..."
echo "temperature_c : frozen cfg ctx=96,k=8,target,guard=96; alpha 0.075/0.10"
echo "pressure_kpa  : frozen cfg ctx=144,k=8,multi,guard=32; alpha 0.05/0.075"
echo "other 4       : EXACT V8"
echo "production    : port 8800 untouched"
echo
PYTHONUNBUFFERED=1 python v9_safe_diagnostic.py | tee v9_safe.log

echo
echo "Done."
echo "Log     : $(pwd)/v9_safe.log"
echo "Metrics : $(pwd)/models/v9_safe_candidate.json"
echo "NOTE    : production 8800/app.py/V8 model pack were not modified."
