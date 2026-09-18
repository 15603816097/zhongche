#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="$ROOT/.venv-full/bin/python"
PORT=8800
PROD_DIR="$ROOT/artifacts/full_arch/production"

mkdir -p "$PROD_DIR"

listener_pid() {
  ss -ltnp 2>/dev/null     | awk -v p=":$PORT" '$4 ~ p"$" {print}'     | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p'     | head -n 1
}

echo "Stopping current 8800 service..."
PID="$(listener_pid || true)"
if [ -n "${PID:-}" ]; then
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.25
  done
  kill -9 "$PID" 2>/dev/null || true
fi

"$PY" - <<'PY'
import socket, time
for _ in range(80):
    s=socket.socket(); s.settimeout(0.2)
    try:
        rc=s.connect_ex(("127.0.0.1",8800))
    finally:
        s.close()
    if rc != 0:
        raise SystemExit(0)
    time.sleep(0.25)
raise SystemExit("port 8800 is still occupied")
PY

echo "Starting exact V8 on 8800..."
PREDICT_WORKERS=2 LGB_INFER_THREADS=8 CALLBACK_TIMEOUT=20 CALLBACK_RETRIES=5 CALLBACK_MIN_AGE=1.0 CALLBACK_GAP=0.25 PYTHONUNBUFFERED=1 nohup "$PY" -m uvicorn app:app   --host 0.0.0.0   --port 8800   --workers 1   > "$PROD_DIR/v8_rollback_8800.log" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PROD_DIR/v8_rollback_8800.pid"

"$PY" - <<'PY'
import json,time,requests
last=None
for _ in range(180):
    try:
        r=requests.get("http://127.0.0.1:8800/health",timeout=2)
        h=r.json()
        print("HEALTH:",json.dumps(h,ensure_ascii=False))
        if (
            r.status_code==200
            and h.get("status")=="ok"
            and h.get("version")=="2.9.0"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
        ):
            print("ROLLBACK PASS: exact V8 is serving on 8800.")
            raise SystemExit(0)
        last=h
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"V8 rollback health timeout: {last}")
PY
