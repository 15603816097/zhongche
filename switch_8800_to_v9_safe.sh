#!/usr/bin/env bash
set -euo pipefail

V9_DIR="/root/v9_lab"
V8_DIR="/root/rail_forecast_v8_deploy"
PY="$V8_DIR/.venv/bin/python"
PROD_PORT=8800
V9_TEST_PORT=8801
PROD_PID_FILE="$V8_DIR/api.pid"
V9_TEST_PID_FILE="$V9_DIR/v9_safe_api.pid"
PROD_LOG="$V9_DIR/v9_safe_prod.log"

cd "$V9_DIR"

echo "[1/6] Verify tested V9-Safe on 8801..."
"$PY" - <<'PY'
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:8801/health', timeout=5) as r:
    h=json.load(r)
print(h)
if h.get('status')!='ok' or h.get('version')!='2.9.0-v9-safe' or not h.get('model_loaded'):
    raise SystemExit('8801 V9-Safe is not healthy')
PY

echo "[2/6] Stop current V8 on 8800..."
OLD_PID=""
if [ -f "$PROD_PID_FILE" ]; then
  OLD_PID="$(cat "$PROD_PID_FILE" 2>/dev/null || true)"
fi
if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
  echo "stopping V8 pid=$OLD_PID"
  kill "$OLD_PID"
  for _ in $(seq 1 20); do
    kill -0 "$OLD_PID" 2>/dev/null || break
    sleep 0.5
  done
fi

# Confirm 8800 is actually free before starting the candidate.
"$PY" - <<'PY'
import socket, time
for _ in range(30):
    s=socket.socket(); s.settimeout(0.2)
    try:
        r=s.connect_ex(('127.0.0.1',8800))
    finally:
        s.close()
    if r != 0:
        raise SystemExit(0)
    time.sleep(0.2)
raise SystemExit('port 8800 is still occupied')
PY

echo "[3/6] Start V9-Safe on production port 8800..."
PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
PYTHONUNBUFFERED=1 \
nohup "$PY" -m uvicorn app_v9_safe:app \
  --host 0.0.0.0 \
  --port 8800 \
  --workers 1 \
  > "$PROD_LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PROD_PID_FILE"
echo "V9-Safe production PID=$NEW_PID"

echo "[4/6] Wait for 8800 V9-Safe health..."
if ! "$PY" - <<'PY'
import json, time, urllib.request
last=None
for _ in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8800/health', timeout=2) as r:
            h=json.load(r)
        print(h)
        if h.get('status')=='ok' and h.get('version')=='2.9.0-v9-safe' and h.get('model_loaded') is True:
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        last=repr(e)
    time.sleep(1)
raise SystemExit(f'V9-Safe 8800 health timeout: {last}')
PY
then
  echo "V9-Safe failed on 8800. Rolling back to V8..." >&2
  kill "$NEW_PID" 2>/dev/null || true
  sleep 2
  cd "$V8_DIR"
  PREDICT_WORKERS=2 LGB_INFER_THREADS=8 PYTHONUNBUFFERED=1 \
  nohup "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8800 --workers 1 > api.log 2>&1 &
  RPID=$!
  echo "$RPID" > "$PROD_PID_FILE"
  "$PY" - <<'PY'
import json,time,urllib.request
for _ in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8800/health',timeout=2) as r:
            h=json.load(r)
        if h.get('status')=='ok' and h.get('version')=='2.9.0' and h.get('ensemble_version')==8:
            print('ROLLBACK V8 HEALTHY:',h)
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass
    time.sleep(1)
raise SystemExit('rollback V8 health timeout')
PY
  exit 1
fi

echo "[5/6] Stop isolated 8801 instance to free resources..."
if [ -f "$V9_TEST_PID_FILE" ]; then
  TPID="$(cat "$V9_TEST_PID_FILE" 2>/dev/null || true)"
  if [ -n "$TPID" ] && kill -0 "$TPID" 2>/dev/null; then
    kill "$TPID" || true
    echo "stopped 8801 pid=$TPID"
  fi
fi

echo "[6/6] Final production health..."
"$PY" - <<'PY'
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:8800/health', timeout=5) as r:
    h=json.load(r)
print('PRODUCTION HEALTH:',h)
assert h.get('status')=='ok'
assert h.get('version')=='2.9.0-v9-safe'
assert h.get('model_loaded') is True
PY

echo
echo "CUTOVER PASS: V9-Safe is now serving on 8800."
echo "Public mapped endpoint remains the same."
echo "Log: $PROD_LOG"
