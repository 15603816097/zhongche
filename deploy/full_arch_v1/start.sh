#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="$ROOT/.runtime/bin/python"
PORT="${PORT:-8800}"
HOST="${HOST:-0.0.0.0}"
PID_FILE="$ROOT/api.pid"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/api.log"

if [ ! -x "$PY" ]; then
  echo "runtime environment missing; run: bash install.sh" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "API already running pid=$OLD_PID"
    exit 0
  fi
fi

PREDICT_WORKERS="${PREDICT_WORKERS:-2}" LGB_INFER_THREADS="${LGB_INFER_THREADS:-8}" CALLBACK_TIMEOUT="${CALLBACK_TIMEOUT:-20}" CALLBACK_RETRIES="${CALLBACK_RETRIES:-5}" CALLBACK_MIN_AGE="${CALLBACK_MIN_AGE:-1.0}" CALLBACK_GAP="${CALLBACK_GAP:-0.25}" PYTHONUNBUFFERED=1 nohup "$PY" -m uvicorn app_full_arch:app   --host "$HOST"   --port "$PORT"   --workers 1   > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "started full-arch pid=$PID port=$PORT"

"$PY" - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = int(sys.argv[1])
last = None
for _ in range(180):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            h = json.load(r)
        if (
            h.get("status") == "ok"
            and h.get("version") == "3.0.0-full-arch"
            and h.get("ensemble_version") == 8
            and h.get("model_loaded") is True
        ):
            print(json.dumps(h, ensure_ascii=False, indent=2))
            print("START PASS")
            raise SystemExit(0)
        last = h
    except SystemExit:
        raise
    except Exception as exc:
        last = repr(exc)
    time.sleep(0.5)
raise SystemExit(f"health timeout: {last}")
PY
