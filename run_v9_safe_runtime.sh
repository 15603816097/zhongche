#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEPLOY_MODELS="${V8_DEPLOY_MODELS:-/root/rail_forecast_v8_deploy/models}"
VENV_PY="${V9_PYTHON:-/root/rail_forecast_v8_deploy/.venv/bin/python}"
PORT="${V9_PORT:-8801}"
PID_FILE="v9_safe_api.pid"
LOG_FILE="v9_safe_api.log"

REQUIRED=(
  model_lgb.pkl
  scaler.pkl
  model_xgb.pkl
  scaler_xgb.pkl
  ensemble_config.pkl
  model_pca_xgb.pkl
  preprocess_pca_xgb.pkl
)

echo "[1/6] Verify V9-Safe offline gate..."
"$VENV_PY" - <<'PY'
import json
from pathlib import Path
p = Path('models/v9_safe_candidate.json')
if not p.is_file():
    raise SystemExit('missing models/v9_safe_candidate.json; run bash run_v9_safe.sh first')
d = json.loads(p.read_text(encoding='utf-8'))
if not d.get('offline_gate_pass'):
    raise SystemExit('V9-Safe offline gate did not pass')
s = d.get('selected') or {}
print('selected temperature alpha:', s.get('temperature_alpha'))
print('selected pressure alpha   :', s.get('pressure_alpha'))
if abs(float(s.get('temperature_alpha')) - 0.075) > 1e-12:
    raise SystemExit('unexpected selected temperature alpha')
if abs(float(s.get('pressure_alpha')) - 0.050) > 1e-12:
    raise SystemExit('unexpected selected pressure alpha')
PY

echo
echo "[2/6] Link exact V8 model pack..."
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

echo
echo "[3/6] Verify production V8 on 8800 is healthy..."
"$VENV_PY" - <<'PY'
import json
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8800/health', timeout=5) as r:
    body = json.load(r)
print('V8 health:', body)
if body.get('ensemble_version') != 8 or not body.get('model_loaded'):
    raise SystemExit('production 8800 is not healthy V8')
PY

echo
echo "[4/6] Syntax check..."
"$VENV_PY" -m py_compile \
  app_v9_safe.py \
  src/v9_safe_runtime.py \
  v9_safe_diagnostic.py

echo
echo "[5/6] Start isolated V9-Safe on port ${PORT}..."
if [ -f "$PID_FILE" ]; then
  old_pid="$(cat "$PID_FILE" || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "stopping previous V9-Safe pid=$old_pid"
    kill "$old_pid" || true
    sleep 2
  fi
fi

PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
PYTHONUNBUFFERED=1 \
nohup "$VENV_PY" -m uvicorn app_v9_safe:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "V9-Safe PID=$(cat "$PID_FILE")"

echo
echo "[6/6] Wait for V9-Safe health..."
"$VENV_PY" - <<PY
import json, time, urllib.request
url = 'http://127.0.0.1:${PORT}/health'
last = None
for _ in range(100):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            body = json.load(r)
        print('V9-Safe health:', body)
        if body.get('status') == 'ok' and body.get('model_loaded') is True:
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        last = repr(e)
    time.sleep(1)
raise SystemExit(f'V9-Safe health timeout: {last}')
PY

echo
echo "V9-Safe isolated runtime is ready."
echo "V8      : http://127.0.0.1:8800"
echo "V9-Safe : http://127.0.0.1:${PORT}"
echo "Log     : $(pwd)/${LOG_FILE}"
echo "PID     : $(pwd)/${PID_FILE}"
echo "Production 8800 was not modified."
