#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

API_PORT="${V83_STRESS_API_PORT:-18884}"
CALLBACK_PORT="${V83_STRESS_CALLBACK_PORT:-18885}"
LOG_DIR="logs"
SERVER_LOG="$LOG_DIR/v83_http_server.log"
STRESS_LOG="$LOG_DIR/v83_http_stress.log"

mkdir -p "$LOG_DIR"

echo
echo "[1/6] Check required files..."
required=(
  "app.py"
  "app_v83.py"
  "stress_v83_http_async.py"
  "src/inference_v83.py"
  "models/v83_final_safe_candidate.json"
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
echo "[2/6] Confirm safe candidate PASS..."
python - <<'PY'
import json
from pathlib import Path
p = Path('models/v83_final_safe_candidate.json')
d = json.loads(p.read_text(encoding='utf-8'))
print('offline_gate_pass :', d.get('offline_gate_pass'))
print('enabled_targets   :', d.get('enabled_targets'))
print('global_rmse_ratio :', d.get('global_rmse_ratio'))
print('proxy_gain_pct    :', d.get('global_proxy_gain_pct'))
if not d.get('offline_gate_pass'):
    raise SystemExit('safe candidate is not PASS')
if 'acoustic_db' in set(d.get('enabled_targets', [])):
    raise SystemExit('acoustic_db must remain exact V8')
PY

echo
echo "[3/6] Confirm verified base app is untouched..."
if ! git diff --exit-code -- app.py >/dev/null; then
  echo "ERROR: app.py has local edits. Refusing stress test."
  git diff -- app.py
  exit 2
fi
echo "app.py working tree unchanged"

echo
echo "[4/6] Syntax + port checks..."
python -m py_compile app_v83.py stress_v83_http_async.py src/inference_v83.py
python - "$API_PORT" "$CALLBACK_PORT" <<'PY'
import socket
import sys
for raw in sys.argv[1:]:
    port = int(raw)
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"port {port} is already in use: {exc}")
    finally:
        s.close()
print("ports available")
PY

echo
echo "[5/6] Start isolated V8.3 service on 127.0.0.1:${API_PORT}..."
: > "$SERVER_LOG"
PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
PYTHONUNBUFFERED=1 \
python -m uvicorn app_v83:app \
  --host 127.0.0.1 \
  --port "$API_PORT" \
  --workers 1 \
  --log-level info \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  set +e
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: V8.3 server exited during startup"
    tail -n 120 "$SERVER_LOG" || true
    exit 2
  fi
  if python - "$API_PORT" <<'PY' >/dev/null 2>&1
import sys
import requests
port = int(sys.argv[1])
r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
raise SystemExit(0 if r.status_code == 200 else 1)
PY
  then
    READY=1
    break
  fi
  sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: V8.3 server did not become ready"
  tail -n 120 "$SERVER_LOG" || true
  exit 2
fi

echo "V8.3 HTTP service ready"
python - "$API_PORT" <<'PY'
import json, sys, requests
port = int(sys.argv[1])
for path in ('/health','/candidate'):
    r = requests.get(f'http://127.0.0.1:{port}{path}', timeout=5)
    print(path, r.status_code, json.dumps(r.json(), ensure_ascii=False))
PY

echo
echo "[6/6] Send 50 async real HTTP requests and verify all callbacks..."
V83_STRESS_API_PORT="$API_PORT" \
V83_STRESS_CALLBACK_PORT="$CALLBACK_PORT" \
V83_STRESS_REQUESTS="${V83_STRESS_REQUESTS:-50}" \
V83_STRESS_SUBMIT_WORKERS="${V83_STRESS_SUBMIT_WORKERS:-10}" \
V83_STRESS_WAIT_SECONDS="${V83_STRESS_WAIT_SECONDS:-240}" \
python stress_v83_http_async.py 2>&1 | tee "$STRESS_LOG"

echo
echo "Server tail (last relevant lines):"
grep -E "V8.3 READY|MODEL READY|PREDICT DONE|CALLBACK (OK|RETRY|FAILED|TERMINAL)|ASYNC (ACCEPTED|READY)|ERROR|Traceback" "$SERVER_LOG" | tail -n 80 || true

echo
echo "Done."
echo "Stress log : $STRESS_LOG"
echo "Server log : $SERVER_LOG"
echo "NOTE: isolated ports only; production 8800 was not touched."
