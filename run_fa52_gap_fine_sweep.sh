#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-/root/zhongche_v8speed}"
PY="${PY:-$SOURCE_ROOT/.venv-v8/bin/python}"
API_PORT=8821
CALLBACK_PORT=8897
OUT="$ROOT/artifacts/fa52_gap_fine"
mkdir -p "$OUT"

if [ ! -x "$PY" ]; then
  echo "missing Python runtime: $PY" >&2
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

BASELINE="$ROOT/artifacts/fa52_speed/baseline_predictions.npz"
if [ ! -f "$BASELINE" ]; then
  echo "missing baseline artifact: $BASELINE" >&2
  echo "run bash run_fa52_speed_sweep.sh first" >&2
  exit 2
fi

stop_port() {
  local port="$1"
  local pid
  pid="$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1)"
  if [ -n "${pid:-}" ]; then
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
for _ in range(180):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
            h=json.load(r)
        if (
            h.get("status")=="ok"
            and h.get("version")=="3.0.0-full-arch"
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

echo "======================================================================"
echo "FA52.895 FINE SWEEP - KEEP ORIGINAL MODEL THREADING"
echo "======================================================================"
echo "baseline prediction artifact : $BASELINE"
echo "production 8800             : untouched"
echo "candidate port              : 127.0.0.1:$API_PORT"
echo "CPU logical                 : $(nproc)"
echo
echo "All configs keep LGB_ESTIMATOR_THREADS=0, XGB=0, PCA_XGB=0."
echo "Only outer workers / LGB executor / callback gap are tested."
echo

CONFIGS=(
  "G_2x8_gap010|2|8|0.10"
  "H_2x8_gap005|2|8|0.05"
  "I_2x8_gap015|2|8|0.15"
  "J_3x8_gap010|3|8|0.10"
  "K_2x6_gap010|2|6|0.10"
  "L_3x6_gap010|3|6|0.10"
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
  echo "PREDICT_WORKERS=$PW LGB_INFER_THREADS=$LGB model_caps=OFF CALLBACK_GAP=$GAP"
  echo "================================================================================================"

  stop_port "$API_PORT"

  PREDICT_WORKERS="$PW" \
  LGB_INFER_THREADS="$LGB" \
  LGB_ESTIMATOR_THREADS=0 \
  XGB_INFER_THREADS=0 \
  PCA_XGB_INFER_THREADS=0 \
  CALLBACK_TIMEOUT=20 \
  CALLBACK_RETRIES=5 \
  CALLBACK_MIN_AGE=1.0 \
  CALLBACK_GAP="$GAP" \
  PYTHONUNBUFFERED=1 \
  nohup "$PY" -m uvicorn app_full_arch:app \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --workers 1 \
    > "$LOG" 2>&1 &

  PID=$!
  echo "candidate pid=$PID"

  if ! wait_health; then
    echo "$NAME | START_FAIL" | tee -a "$SUMMARY"
    tail -n 100 "$LOG" || true
    stop_port "$API_PORT"
    continue
  fi

  set +e
  API_PORT="$API_PORT" \
  CALLBACK_PORT="$CALLBACK_PORT" \
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
  echo "$NAME | $RESULT" | tee -a "$SUMMARY"

  stop_port "$API_PORT"
  sleep 1
done

echo
echo "================================================================================================"
echo "FA52 GAP FINE SWEEP SUMMARY"
echo "================================================================================================"
cat "$SUMMARY"
echo
echo "Production 8800 was never stopped or modified."
