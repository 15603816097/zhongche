#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-/root/zhongche_v8speed}"
PY="${PY:-$SOURCE_ROOT/.venv-v8/bin/python}"
API_PORT=8821
CALLBACK_PORT=8897
OUT="$ROOT/artifacts/fa52_speed"
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
echo "FA52.895 EXACT-PREDICTION RUNTIME SWEEP"
echo "======================================================================"
echo "workspace      : $ROOT"
echo "production 8800: untouched"
echo "candidate port : 127.0.0.1:$API_PORT"
echo "CPU logical    : $(nproc)"
echo

echo "[0] Capture exact frozen Full Architecture baseline..."
LGB_INFER_THREADS=8 \
LGB_ESTIMATOR_THREADS=0 \
XGB_INFER_THREADS=0 \
PCA_XGB_INFER_THREADS=0 \
"$PY" capture_fa52_baseline.py

CONFIGS=(
  "A_original_2x8_gap025|2|8|0|0|0|0.25"
  "B_4x4_cap1_gap025|4|4|1|1|1|0.25"
  "C_4x2_cap1_gap010|4|2|1|1|1|0.10"
  "D_5x2_cap1_gap010|5|2|1|1|1|0.10"
  "E_6x2_cap1_gap010|6|2|1|1|1|0.10"
  "F_6x1_cap1_gap010|6|1|1|1|1|0.10"
)

SUMMARY="$OUT/sweep_summary.txt"
: > "$SUMMARY"

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME PW LGB LGBEST XGB PCA GAP <<< "$row"
  LOG="$OUT/${NAME}_api.log"
  TESTLOG="$OUT/${NAME}_test.log"

  echo
  echo "================================================================================================"
  echo "CONFIG $NAME"
  echo "PREDICT_WORKERS=$PW LGB_INFER_THREADS=$LGB LGB_ESTIMATOR_THREADS=$LGBEST XGB=$XGB PCA_XGB=$PCA CALLBACK_GAP=$GAP"
  echo "================================================================================================"

  stop_port "$API_PORT"

  PREDICT_WORKERS="$PW" \
  LGB_INFER_THREADS="$LGB" \
  LGB_ESTIMATOR_THREADS="$LGBEST" \
  XGB_INFER_THREADS="$XGB" \
  PCA_XGB_INFER_THREADS="$PCA" \
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
echo "FA52 SPEED SWEEP SUMMARY"
echo "================================================================================================"
cat "$SUMMARY"
echo
echo "Production 8800 was never stopped or modified."
