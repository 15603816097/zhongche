#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv-full/bin/python"
API_PORT=8810
LOG_DIR="artifacts/full_arch/runtime"
LOG_FILE="$LOG_DIR/full_arch_api_stage7.log"
PID_FILE="$LOG_DIR/full_arch_api_stage7.pid"
HEALTH_FILE="/tmp/full_arch_stage7_health.json"

if [ ! -x "$PY" ]; then
  echo "missing .venv-full; run Stage 1 first" >&2
  exit 2
fi
if [ ! -f models/full_arch/dynamic_gate_config.json ]; then
  echo "missing Stage 5 gate config; run Stage 5 first" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
rm -f "$HEALTH_FILE"

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$API_PORT$"; then
  echo "port $API_PORT is already in use; refusing to kill an unknown process" >&2
  exit 2
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]8811$"; then
  echo "port 8811 is already in use; callback receiver needs it" >&2
  exit 2
fi

cleanup() {
  if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "FULL ARCHITECTURE - STAGE 7 ISOLATED API"
echo "============================================================"
echo "production V8 on port 8800 is NOT touched"
echo "full-arch test API  : 127.0.0.1:$API_PORT"
echo "log                 : $LOG_FILE"

PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
CALLBACK_TIMEOUT=20 \
CALLBACK_RETRIES=5 \
CALLBACK_MIN_AGE=1.0 \
CALLBACK_GAP=0.25 \
PYTHONUNBUFFERED=1 \
nohup "$PY" -m uvicorn app_full_arch:app \
  --host 127.0.0.1 \
  --port "$API_PORT" \
  --workers 1 \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

# Do not depend on curl being installed. Use the project Python environment
# and requests, which Stage 1 explicitly installs.
ready=0
for _ in $(seq 1 180); do
  if "$PY" - "$API_PORT" "$HEALTH_FILE" <<'PY'
import json
import sys
import requests

port = int(sys.argv[1])
path = sys.argv[2]
try:
    r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
    if r.status_code != 200:
        raise SystemExit(1)
    data = r.json()
    if data.get("status") != "ok":
        raise SystemExit(1)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
except Exception:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi

  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "API process exited during startup" >&2
    tail -n 100 "$LOG_FILE" >&2 || true
    exit 2
  fi
  sleep 0.5
done

if [ "$ready" -ne 1 ]; then
  echo "API did not become healthy in time" >&2
  echo "[port state]" >&2
  ss -ltnp 2>/dev/null | grep -E "[:.]$API_PORT\b" >&2 || true
  echo "[last API log lines]" >&2
  tail -n 100 "$LOG_FILE" >&2 || true
  exit 2
fi

echo
echo "[health]"
"$PY" - "$HEALTH_FILE" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
PY
echo

"$PY" test_full_arch_stage7_callback_stress.py

echo
echo "[last API log lines]"
tail -n 30 "$LOG_FILE" || true

echo
echo "STAGE 7 PASS"
