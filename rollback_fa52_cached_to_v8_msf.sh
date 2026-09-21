#!/usr/bin/env bash
set -euo pipefail

FA_ROOT="$(cd "$(dirname "$0")" && pwd)"
V8_ROOT="${V8_ROOT:-/root/zhongche_v8speed}"
PORT="${PORT:-8800}"
PY="${PY:-$V8_ROOT/.venv-v8/bin/python}"
LOG_DIR="$V8_ROOT/logs"
LOG_FILE="$LOG_DIR/v8_msf_rollback_8800.log"

listener_pid() {
  ss -ltnp 2>/dev/null \
    | awk -v p=":$PORT" '$4 ~ p"$" {print}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -n 1
}

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

mkdir -p "$LOG_DIR"
cd "$V8_ROOT"

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

echo $! > "$V8_ROOT/v8_msf_rollback_8800.pid"

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
            and h.get("version")=="2.9.0-v8-msf"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
            and h.get("multiscale_fusion") is True
        ):
            print("V8-MSF ROLLBACK: PASS")
            raise SystemExit(0)
        last=h
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"rollback health timeout: {last}")
PY
