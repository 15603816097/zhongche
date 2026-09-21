#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8800}"
PY="$ROOT/.venv-v8/bin/python"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/v8_msf_8800.log"
PID_FILE="$ROOT/v8_msf_8800.pid"

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
echo "SWITCH PRODUCTION 8800 TO V8-MSF"
echo "======================================================================"

echo "[1/5] Verify multi-scale candidate..."
"$PY" - <<'PY'
import json
from pathlib import Path
p=Path("models/v8_multiscale_fusion_candidate.json")
if not p.is_file():
    raise SystemExit(f"missing {p}")
d=json.loads(p.read_text(encoding="utf-8"))
if not d.get("offline_gate_pass"):
    raise SystemExit("offline gate did not pass")
enabled=d.get("enabled_targets") or []
expected={"temperature_c","current_a","speed_rpm","acoustic_db"}
if set(enabled) != expected:
    raise SystemExit(f"unexpected enabled targets: {enabled}")
print("enabled targets:", enabled)
print("flat RMSE ratio:", d.get("global",{}).get("flat_rmse_ratio"))
print("proxy gain:", d.get("global",{}).get("global_proxy_gain"))
print("trend gain:", d.get("global",{}).get("global_trend_gain"))
print("candidate config: PASS")
PY

echo
echo "[2/5] Syntax check..."
"$PY" -m py_compile app_v8_msf.py src/v8_msf_runtime.py

echo
echo "[3/5] Stop current listener on $PORT..."
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
echo "[4/5] Start V8-MSF production..."
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
nohup "$PY" -m uvicorn app_v8_msf:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "pid=$NEW_PID"

echo
echo "[5/5] Health check..."
"$PY" - "$PORT" <<'PY'
import json,sys,time,urllib.request
port=int(sys.argv[1])
last=None
for _ in range(180):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
            h=json.load(r)
        print(json.dumps(h,ensure_ascii=False))
        enabled=set(h.get("multiscale_enabled_targets") or [])
        if (
            h.get("status")=="ok"
            and h.get("version")=="2.9.0-v8-msf"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
            and h.get("multiscale_fusion") is True
            and enabled=={"temperature_c","current_a","speed_rpm","acoustic_db"}
            and h.get("predict_workers")==6
            and h.get("lgb_infer_threads")==2
            and abs(float(h.get("callback_gap",999))-0.10) < 1e-12
        ):
            print("V8-MSF PRODUCTION: PASS")
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
