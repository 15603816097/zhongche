#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8800}"
PY="$ROOT/.venv-v8/bin/python"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/v8_conservative_8800.log"

listener_pid() {
  ss -ltnp 2>/dev/null \
    | awk -v p=":$PORT" '$4 ~ p"$" {print}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -n 1
}

OLD_PID="$(listener_pid || true)"
if [ -n "${OLD_PID:-}" ]; then
  kill "$OLD_PID" 2>/dev/null || true
  sleep 2
  kill -9 "$OLD_PID" 2>/dev/null || true
fi

mkdir -p "$LOG_DIR"
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
LGB_ESTIMATOR_THREADS=1 \
XGB_INFER_THREADS=1 \
PCA_XGB_INFER_THREADS=1 \
CALLBACK_TIMEOUT=20 \
CALLBACK_RETRIES=5 \
CALLBACK_MIN_AGE=1.0 \
CALLBACK_GAP=0.25 \
PYTHONUNBUFFERED=1 \
nohup "$PY" -m uvicorn app:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  > "$LOG_FILE" 2>&1 &

echo $! > "$ROOT/v8_conservative_8800.pid"

"$PY" - "$PORT" <<'PY'
import json,sys,time,urllib.request
port=int(sys.argv[1])
last=None
for _ in range(120):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
            h=json.load(r)
        print(json.dumps(h,ensure_ascii=False))
        if (
            h.get("status")=="ok"
            and h.get("version")=="2.9.0"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
            and h.get("predict_workers")==2
            and h.get("lgb_infer_threads")==8
            and abs(float(h.get("callback_gap",999))-0.25) < 1e-12
        ):
            print("V8 CONSERVATIVE ROLLBACK: PASS")
            raise SystemExit(0)
        last=h
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"rollback health timeout: {last}")
PY
