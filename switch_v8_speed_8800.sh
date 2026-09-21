#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8800}"
PY="$ROOT/.venv-v8/bin/python"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/v8_speed_8800.log"
PID_FILE="$ROOT/v8_speed_8800.pid"

if [ ! -x "$PY" ]; then
  echo "missing runtime: $PY" >&2
  exit 2
fi

listener_pid() {
  ss -ltnp 2>/dev/null \
    | awk -v p=":$PORT" '$4 ~ p"$" {print}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -n 1
}

wait_port_free() {
  "$PY" - "$PORT" <<'PY'
import socket,sys,time
port=int(sys.argv[1])
for _ in range(80):
    s=socket.socket(); s.settimeout(0.2)
    try:
        rc=s.connect_ex(("127.0.0.1",port))
    finally:
        s.close()
    if rc != 0:
        raise SystemExit(0)
    time.sleep(0.25)
raise SystemExit(f"port {port} is still occupied")
PY
}

echo "======================================================================"
echo "SWITCH PRODUCTION 8800 TO V8-SPEED"
echo "======================================================================"
echo "PREDICT_WORKERS=6"
echo "LGB_INFER_THREADS=2"
echo "LGB_ESTIMATOR_THREADS=1"
echo "XGB_INFER_THREADS=1"
echo "PCA_XGB_INFER_THREADS=1"
echo "CALLBACK_GAP=0.10"
echo

echo "[1/4] Verify V8 model config..."
"$PY" - <<'PY'
import pickle
from pathlib import Path
p=Path("models/ensemble_config.pkl")
with p.open("rb") as f:
    c=pickle.load(f)
print("version:", c.get("version"))
print("trajectory_model:", c.get("trajectory_model"))
if int(c.get("version",-1)) != 8:
    raise SystemExit("not V8")
if str(c.get("trajectory_model","")) != "pca_xgb_source_aware_hf_v1":
    raise SystemExit("unexpected trajectory_model")
print("V8 config: PASS")
PY

echo
echo "[2/4] Stop current listener on $PORT..."
OLD_PID="$(listener_pid || true)"
if [ -n "${OLD_PID:-}" ]; then
  echo "stopping pid=$OLD_PID"
  kill "$OLD_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$OLD_PID" 2>/dev/null || break
    sleep 0.25
  done
  kill -9 "$OLD_PID" 2>/dev/null || true
fi
wait_port_free
echo "port $PORT: FREE"

echo
echo "[3/4] Start V8-Speed..."
mkdir -p "$LOG_DIR"
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PREDICT_WORKERS=6 \
LGB_INFER_THREADS=2 \
LGB_ESTIMATOR_THREADS=1 \
XGB_INFER_THREADS=1 \
PCA_XGB_INFER_THREADS=1 \
CALLBACK_TIMEOUT=20 \
CALLBACK_RETRIES=5 \
CALLBACK_MIN_AGE=1.0 \
CALLBACK_GAP=0.10 \
PYTHONUNBUFFERED=1 \
nohup "$PY" -m uvicorn app:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "pid=$NEW_PID"

echo
echo "[4/4] Health check..."
"$PY" - "$PORT" <<'PY'
import json,sys,time,urllib.request
port=int(sys.argv[1])
last=None
for _ in range(180):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
            h=json.load(r)
        print(json.dumps(h,ensure_ascii=False))
        if (
            h.get("status")=="ok"
            and h.get("version")=="2.9.0"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
            and h.get("predict_workers")==6
            and h.get("lgb_infer_threads")==2
            and abs(float(h.get("callback_gap",999))-0.10) < 1e-12
        ):
            print("V8-SPEED PRODUCTION: PASS")
            raise SystemExit(0)
        last=h
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"health timeout: {last}")
PY

echo
echo "Production URL: http://180.127.11.177:24188/predict"
echo "Health URL    : http://180.127.11.177:24188/health"
echo "Log           : $LOG_FILE"
