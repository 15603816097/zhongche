#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

API_PORT="${V81_STRESS_API_PORT:-18882}"
CALLBACK_PORT="${V81_STRESS_CALLBACK_PORT:-18883}"
LOG_DIR="logs"
SERVER_LOG="$LOG_DIR/v81_http_server.log"
STRESS_LOG="$LOG_DIR/v81_http_stress.log"

mkdir -p "$LOG_DIR"

echo
echo "[1/5] Check required files..."
required=(
  "app.py"
  "app_v81.py"
  "stress_v81_http_async.py"
  "src/inference_v81.py"
  "src/deep/patchtst_temperature_runtime.py"
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
echo "[2/5] Confirm verified base app is untouched..."
if ! git diff --exit-code -- app.py >/dev/null; then
  echo "ERROR: app.py has local edits. Refusing stress test."
  git diff -- app.py
  exit 2
fi
echo "app.py working tree unchanged"

echo
echo "[3/5] Syntax + port checks..."
python -m py_compile app_v81.py stress_v81_http_async.py
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
echo "[4/5] Start isolated real HTTP V8.1 service on 127.0.0.1:${API_PORT}..."
: > "$SERVER_LOG"
V81_TEMPERATURE_ENABLED=1 \
V81_TEMPERATURE_WEIGHT=0.15 \
python -m uvicorn app_v81:app \
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
    echo "ERROR: V8.1 server exited during startup"
    tail -n 100 "$SERVER_LOG" || true
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
  echo "ERROR: V8.1 server did not become ready"
  tail -n 100 "$SERVER_LOG" || true
  exit 2
fi

echo "V8.1 HTTP service ready"

echo
echo "[5/5] Send 50 async real HTTP requests and verify all callbacks..."
V81_STRESS_API_PORT="$API_PORT" \
V81_STRESS_CALLBACK_PORT="$CALLBACK_PORT" \
V81_STRESS_REQUESTS="${V81_STRESS_REQUESTS:-50}" \
V81_STRESS_SUBMIT_WORKERS="${V81_STRESS_SUBMIT_WORKERS:-10}" \
V81_STRESS_WAIT_SECONDS="${V81_STRESS_WAIT_SECONDS:-240}" \
python stress_v81_http_async.py 2>&1 | tee "$STRESS_LOG"

echo
echo "Server tail (last relevant lines):"
grep -E "MODEL READY|PREDICT DONE|CALLBACK (OK|RETRY|FAILED|TERMINAL)|ASYNC (ACCEPTED|READY)|ERROR|Traceback" "$SERVER_LOG" | tail -n 60 || true

echo
echo "Done."
echo "Stress log : $STRESS_LOG"
echo "Server log : $SERVER_LOG"
echo "NOTE: this used isolated ports only; app.py / ensemble_config.pkl / verified callback schema were not modified."
