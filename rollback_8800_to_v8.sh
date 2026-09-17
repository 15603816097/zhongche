#!/usr/bin/env bash
set -euo pipefail

V8_DIR="/root/rail_forecast_v8_deploy"
PY="$V8_DIR/.venv/bin/python"
PID_FILE="$V8_DIR/api.pid"
LOG_FILE="$V8_DIR/api.log"

CURRENT_PID=""
if [ -f "$PID_FILE" ]; then
  CURRENT_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
  echo "Stopping current 8800 service pid=$CURRENT_PID"
  kill "$CURRENT_PID" || true
  for _ in $(seq 1 30); do
    kill -0 "$CURRENT_PID" 2>/dev/null || break
    sleep 0.5
  done
fi

"$PY" - <<'PY'
import socket, time
for _ in range(40):
    s = socket.socket()
    s.settimeout(0.2)
    try:
        rc = s.connect_ex(('127.0.0.1', 8800))
    finally:
        s.close()
    if rc != 0:
        raise SystemExit(0)
    time.sleep(0.25)
raise SystemExit('port 8800 is still occupied')
PY

cd "$V8_DIR"
PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
PYTHONUNBUFFERED=1 \
nohup "$PY" -m uvicorn app:app \
  --host 0.0.0.0 \
  --port 8800 \
  --workers 1 \
  > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "Started V8 pid=$NEW_PID"

"$PY" - <<'PY'
import json, time, urllib.request
last = None
for _ in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8800/health', timeout=2) as r:
            h = json.load(r)
        print('HEALTH:', h)
        if (
            h.get('status') == 'ok'
            and h.get('version') == '2.9.0'
            and h.get('ensemble_version') == 8
            and h.get('model_loaded') is True
        ):
            print('ROLLBACK PASS: V8 is serving on 8800.')
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        last = repr(exc)
    time.sleep(1)
raise SystemExit(f'V8 rollback health timeout: {last}')
PY
