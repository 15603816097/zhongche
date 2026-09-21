#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-/root/zhongche_v8speed}"
PY="${PY:-$SOURCE_ROOT/.venv-v8/bin/python}"
PORT="${PORT:-8800}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/fa52_cached_8800.log"
PID_FILE="$ROOT/fa52_cached_8800.pid"

if [ ! -x "$PY" ]; then
  echo "missing Python runtime: $PY" >&2
  exit 2
fi

if [ ! -e "$ROOT/data" ]; then
  ln -s "$SOURCE_ROOT/data" "$ROOT/data"
fi

mkdir -p "$ROOT/models" "$LOG_DIR"
for name in \
  model_lgb.pkl \
  scaler.pkl \
  model_xgb.pkl \
  scaler_xgb.pkl \
  ensemble_config.pkl \
  model_pca_xgb.pkl \
  preprocess_pca_xgb.pkl
do
  if [ ! -e "$ROOT/models/$name" ]; then
    ln -s "$SOURCE_ROOT/models/$name" "$ROOT/models/$name"
  fi
done

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
echo "SWITCH PRODUCTION 8800 TO FA52.895 CACHED"
echo "======================================================================"

echo "[1/5] Verify frozen Full Architecture config..."
"$PY" - <<'PY'
import json
from pathlib import Path
p=Path("full_arch_frozen_gate_v1.json")
d=json.loads(p.read_text(encoding="utf-8"))
if d.get("global_gate_pass") is not True:
    raise SystemExit("global_gate_pass is not true")
if d.get("enabled_targets") != ["speed_rpm","acoustic_db","pressure_kpa"]:
    raise SystemExit(f"unexpected enabled_targets={d.get('enabled_targets')}")
print("frozen config: PASS")
PY

echo
echo "[2/5] Syntax check..."
"$PY" -m py_compile app_full_arch.py src/full_arch_runtime.py src/inference.py src/v8_runtime.py

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
echo "[4/5] Start cached FA52.895 production..."
env -u FULL_ARCH_GATE_CONFIG \
PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
LGB_ESTIMATOR_THREADS=0 \
XGB_INFER_THREADS=0 \
PCA_XGB_INFER_THREADS=0 \
CALLBACK_TIMEOUT=20 \
CALLBACK_RETRIES=5 \
CALLBACK_MIN_AGE=1.0 \
CALLBACK_GAP=0.25 \
PYTHONUNBUFFERED=1 \
nohup "$PY" -m uvicorn app_full_arch:app \
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
        if (
            h.get("status")=="ok"
            and h.get("version")=="3.0.0-full-arch"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
            and h.get("predict_workers")==2
            and h.get("lgb_infer_threads")==8
            and abs(float(h.get("callback_gap",999))-0.25) < 1e-12
        ):
            print("FA52.895 CACHED PRODUCTION: PASS")
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
