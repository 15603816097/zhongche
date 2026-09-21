#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="$ROOT/.venv-v8/bin/python"
BASE_PORT=8800
API_PORT=8810
CALLBACK_PORT=8899
OUT="$ROOT/artifacts/v8_speed"
mkdir -p "$OUT"

if [ ! -x "$PY" ]; then
  echo "missing $PY; deploy the V8 baseline first" >&2
  exit 2
fi

echo "======================================================================"
echo "V8-SPEED ISOLATED RUNTIME SWEEP"
echo "======================================================================"
echo "baseline       : 127.0.0.1:$BASE_PORT (untouched)"
echo "candidate port : 127.0.0.1:$API_PORT"
echo "CPU logical    : $(nproc)"
echo "branch HEAD    : $(git rev-parse HEAD)"
echo

"$PY" - <<'PY'
import json, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8800/health", timeout=5) as r:
    h=json.load(r)
print("baseline health:", h)
assert h.get("status")=="ok"
assert h.get("version")=="2.9.0"
assert h.get("ensemble_version")==8
assert h.get("model_loaded") is True
PY

stop_candidate() {
  local pid="${1:-}"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

wait_health() {
  "$PY" - "$API_PORT" <<'PY'
import json,sys,time,urllib.request
port=int(sys.argv[1])
last=None
for _ in range(120):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
            h=json.load(r)
        if (
            h.get("status")=="ok"
            and h.get("version")=="2.9.0"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
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
}

CONFIGS=(
  "A_2x8_gap025|2|8|0.25"
  "B_4x4_gap025|4|4|0.25"
  "C_4x4_gap010|4|4|0.10"
  "D_6x2_gap010|6|2|0.10"
  "E_8x1_gap010|8|1|0.10"
)

SUMMARY="$OUT/sweep_summary.txt"
: > "$SUMMARY"

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME PW LGB GAP <<< "$row"
  LOG="$OUT/${NAME}_api.log"
  TESTLOG="$OUT/${NAME}_test.log"

  echo
  echo "================================================================================================"
  echo "CONFIG $NAME"
  echo "PREDICT_WORKERS=$PW LGB_INFER_THREADS=$LGB LGB_ESTIMATOR_THREADS=1 XGB=1 PCA_XGB=1 CALLBACK_GAP=$GAP"
  echo "================================================================================================"

  oldpid="$(ss -ltnp 2>/dev/null | awk -v p=":$API_PORT" '$4 ~ p"$" {print}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1)"
  stop_candidate "$oldpid"

  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PREDICT_WORKERS="$PW" \
  LGB_INFER_THREADS="$LGB" \
  LGB_ESTIMATOR_THREADS=1 \
  XGB_INFER_THREADS=1 \
  PCA_XGB_INFER_THREADS=1 \
  CALLBACK_TIMEOUT=20 \
  CALLBACK_RETRIES=5 \
  CALLBACK_MIN_AGE=1.0 \
  CALLBACK_GAP="$GAP" \
  PYTHONUNBUFFERED=1 \
  nohup "$PY" -m uvicorn app:app \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --workers 1 \
    > "$LOG" 2>&1 &

  PID=$!
  echo "candidate pid=$PID"

  if ! wait_health; then
    echo "$NAME | START_FAIL" | tee -a "$SUMMARY"
    tail -n 80 "$LOG" || true
    stop_candidate "$PID"
    continue
  fi

  set +e
  BASE_PORT="$BASE_PORT" \
  API_PORT="$API_PORT" \
  CALLBACK_PORT="$CALLBACK_PORT" \
  N_REQUESTS=50 \
  WAIT_TIMEOUT=600 \
  "$PY" test_v8_speed_candidate.py | tee "$TESTLOG"
  RC=${PIPESTATUS[0]}
  set -e

  RESULT="$(grep '^RESULT ' "$TESTLOG" | tail -n1 || true)"
  if [ -z "$RESULT" ]; then
    RESULT="NO_RESULT rc=$RC"
  fi
  echo "$NAME | $RESULT" | tee -a "$SUMMARY"

  stop_candidate "$PID"
  sleep 1
done

echo
echo "================================================================================================"
echo "SWEEP SUMMARY"
echo "================================================================================================"
cat "$SUMMARY"
echo
echo "Baseline 8800 was never stopped or modified."
echo "Do not switch production yet; use the fastest PASS result for the next step."
