#!/usr/bin/env bash
set -u -o pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="$ROOT/.venv-v8/bin/python"
BASE_PORT=8800
API_PORT=8810
CALLBACK_PORT=8899
REPEATS="${REPEATS:-3}"
OUT="$ROOT/artifacts/v8_speed_confirm"
mkdir -p "$OUT"

if [ ! -x "$PY" ]; then
  echo "missing $PY" >&2
  exit 2
fi

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
  "F_5x2_gap010|5|2|0.10"
  "D_6x2_gap010|6|2|0.10"
  "H_6x2_gap005|6|2|0.05"
)

CSV="$OUT/results.csv"
echo "config,run,pass,max_diff,total_seconds,throughput,accepted,callbacks,duplicates,errors" > "$CSV"

echo "======================================================================"
echo "V8-SPEED REPEAT CONFIRMATION"
echo "======================================================================"
echo "CPU logical : $(nproc)"
echo "repeats     : $REPEATS"
echo "baseline    : 127.0.0.1:$BASE_PORT untouched"
echo

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME PW LGB GAP <<< "$row"

  for run in $(seq 1 "$REPEATS"); do
    echo
    echo "------------------------------------------------------------------------"
    echo "$NAME run $run/$REPEATS"
    echo "PREDICT_WORKERS=$PW LGB_INFER_THREADS=$LGB CALLBACK_GAP=$GAP"
    echo "------------------------------------------------------------------------"

    oldpid="$(ss -ltnp 2>/dev/null | awk -v p=":$API_PORT" '$4 ~ p"$" {print}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1)"
    stop_candidate "$oldpid"

    LOG="$OUT/${NAME}_run${run}_api.log"
    TESTLOG="$OUT/${NAME}_run${run}_test.log"

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
    if ! wait_health; then
      echo "$NAME,$run,0,nan,nan,nan,0,0,0,1" >> "$CSV"
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
    if [ -n "$RESULT" ]; then
      "$PY" - "$NAME" "$run" "$RESULT" "$CSV" <<'PY'
import re,sys
name,run,line,csv=sys.argv[1:5]
pairs=dict(re.findall(r'([a-z_]+)=([^ ]+)', line))
with open(csv,'a',encoding='utf-8') as f:
    f.write(','.join([
        name,run,
        pairs.get('pass','0'),
        pairs.get('max_diff','nan'),
        pairs.get('total_seconds','nan'),
        pairs.get('throughput','nan'),
        pairs.get('accepted','0'),
        pairs.get('callbacks','0'),
        pairs.get('duplicates','0'),
        pairs.get('errors','1'),
    ])+'\n')
PY
    else
      echo "$NAME,$run,0,nan,nan,nan,0,0,0,1" >> "$CSV"
    fi

    stop_candidate "$PID"
    sleep 1
  done
done

echo
echo "======================================================================"
echo "REPEAT CONFIRMATION SUMMARY"
echo "======================================================================"
"$PY" - "$CSV" <<'PY'
import csv,statistics,sys
path=sys.argv[1]
rows=list(csv.DictReader(open(path,encoding='utf-8')))
names=[]
for r in rows:
    if r['config'] not in names:
        names.append(r['config'])
for name in names:
    rr=[r for r in rows if r['config']==name]
    good=[r for r in rr if r['pass']=='1']
    times=[float(r['total_seconds']) for r in good]
    thr=[float(r['throughput']) for r in good]
    diffs=[float(r['max_diff']) for r in good]
    if not times:
        print(f"{name}: PASS 0/{len(rr)}")
        continue
    print(
        f"{name}: PASS {len(good)}/{len(rr)} "
        f"median={statistics.median(times):.6f}s "
        f"mean={statistics.mean(times):.6f}s "
        f"min={min(times):.6f}s max={max(times):.6f}s "
        f"median_throughput={statistics.median(thr):.6f} "
        f"max_diff={max(diffs):.3e}"
    )
PY

echo
echo "CSV: $CSV"
echo "Production 8800 was never modified."
