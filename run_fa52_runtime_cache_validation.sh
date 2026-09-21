#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-/root/zhongche_v8speed}"
PY="${PY:-$SOURCE_ROOT/.venv-v8/bin/python}"
PORT=8823
LOG_DIR="$ROOT/artifacts/fa52_runtime_cache"
LOG="$LOG_DIR/api.log"
BASELINE="$ROOT/artifacts/fa52_speed/baseline_predictions.npz"

mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "FA52.895 EXACT RUNTIME-CACHE VALIDATION"
echo "======================================================================"
echo "production 8800 : untouched"
echo "candidate port  : 127.0.0.1:$PORT"
echo "baseline config : frozen Full Architecture V1"
echo "runtime config  : original 2x8 / callback gap 0.25"
echo

if [ ! -x "$PY" ]; then
  echo "missing Python runtime: $PY" >&2
  exit 2
fi

if [ ! -f "$BASELINE" ]; then
  echo "missing baseline artifact: $BASELINE" >&2
  exit 2
fi

if [ ! -e "$ROOT/data" ]; then
  ln -s "$SOURCE_ROOT/data" "$ROOT/data"
fi

mkdir -p "$ROOT/models"
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
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_port

echo "[1/2] Start optimized frozen FA52 candidate..."
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
        if (
            h.get("status")=="ok"
            and h.get("version")=="3.0.0-full-arch"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
            and h.get("predict_workers")==2
            and h.get("lgb_infer_threads")==8
            and abs(float(h.get("callback_gap",999))-0.25) < 1e-12
        ):
            print("candidate health: PASS")
            raise SystemExit(0)
        last=h
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"candidate health timeout: {last}")
PY

echo
echo "[2/2] Three exact-output callback stress repeats..."
SUMMARY="$LOG_DIR/summary.txt"
: > "$SUMMARY"

PORTS=(8895 8894 8893)
for i in 1 2 3; do
  TESTLOG="$LOG_DIR/repeat_${i}.log"
  echo
  echo "--------------------------------------------------------------------"
  echo "REPEAT $i"
  echo "--------------------------------------------------------------------"

  set +e
  API_PORT="$PORT" \
  CALLBACK_PORT="${PORTS[$((i-1))]}" \
  N_REQUESTS=50 \
  WAIT_TIMEOUT=900 \
  BASELINE_PATH="$BASELINE" \
  "$PY" test_fa52_speed_candidate.py | tee "$TESTLOG"
  RC=${PIPESTATUS[0]}
  set -e

  RESULT="$(grep '^RESULT ' "$TESTLOG" | tail -n1 || true)"
  if [ -z "$RESULT" ]; then
    RESULT="NO_RESULT rc=$RC"
  fi
  echo "repeat_$i | $RESULT" | tee -a "$SUMMARY"

  if [ "$RC" -ne 0 ]; then
    echo "repeat $i failed" >&2
    stop_port
    exit "$RC"
  fi
  sleep 1
done

echo
echo "======================================================================"
echo "FA52 RUNTIME-CACHE SUMMARY"
echo "======================================================================"
cat "$SUMMARY"

"$PY" - "$SUMMARY" <<'PY'
import re, statistics, sys
text=open(sys.argv[1],encoding="utf-8").read()
vals=[float(x) for x in re.findall(r"total_seconds=([0-9.]+)", text)]
if len(vals)!=3:
    raise SystemExit(f"expected 3 totals, got {vals}")
print(f"median seconds  : {statistics.median(vals):.6f}")
print(f"best seconds    : {min(vals):.6f}")
print(f"worst seconds   : {max(vals):.6f}")
print("old A reference : 19.513674")
print(
    "median speedup  : "
    f"{(19.513674/statistics.median(vals)-1.0)*100:+.2f}%"
)
PY

stop_port
echo
echo "Production 8800 was never stopped or modified."
