#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-/root/zhongche_v8speed}"
PY="${PY:-$SOURCE_ROOT/.venv-v8/bin/python}"
PORT=8822
CALLBACK_PORT=8896
LOG="$ROOT/artifacts/fa52_acoustic075/api.log"

echo "======================================================================"
echo "FA52.895 + ACOUSTIC GATE x0.75 ONLINE VALIDATION"
echo "======================================================================"
echo "production 8800: untouched"
echo "candidate port : 127.0.0.1:$PORT"
echo

if [ ! -x "$PY" ]; then
  echo "missing Python runtime: $PY" >&2
  exit 2
fi

if [ ! -e "$ROOT/data" ]; then
  ln -s "$SOURCE_ROOT/data" "$ROOT/data"
fi

mkdir -p "$ROOT/models" "$ROOT/artifacts/fa52_acoustic075"
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

stop_port() {
  local pid
  pid="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1)"
  if [ -n "${pid:-}" ]; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
}

echo "[1/3] Capture expected candidate from frozen FA52 baseline..."
FULL_ARCH_GATE_CONFIG="" \
LGB_INFER_THREADS=8 \
LGB_ESTIMATOR_THREADS=0 \
XGB_INFER_THREADS=0 \
PCA_XGB_INFER_THREADS=0 \
"$PY" capture_fa52_acoustic075_expected.py

echo
echo "[2/3] Start isolated candidate..."
stop_port

FULL_ARCH_GATE_CONFIG="full_arch_gate_acoustic075.json" \
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
nohup "$PY" -m uvicorn app_fa52_acoustic075:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  --workers 1 \
  > "$LOG" 2>&1 &

PID=$!
echo "candidate pid=$PID"

"$PY" - "$PORT" <<'PY'
import json,sys,time,urllib.request
port=int(sys.argv[1])
last=None
for _ in range(180):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
            h=json.load(r)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/candidate",timeout=2) as r:
            c=json.load(r)
        if (
            h.get("status")=="ok"
            and h.get("version")=="3.0.1-fa52-acoustic075"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
            and c.get("shrink_scales",{}).get("acoustic_db")==0.75
        ):
            print("candidate health: PASS")
            raise SystemExit(0)
        last=(h,c)
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"candidate health timeout: {last}")
PY

echo
echo "[3/3] Exact reproduction + callback stress..."
set +e
API_PORT="$PORT" \
CALLBACK_PORT="$CALLBACK_PORT" \
N_REQUESTS=50 \
WAIT_TIMEOUT=900 \
"$PY" test_fa52_acoustic075_online.py
RC=$?
set -e

stop_port
exit "$RC"
