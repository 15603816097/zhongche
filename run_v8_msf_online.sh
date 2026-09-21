#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="$ROOT/.venv-v8/bin/python"
PORT=8811
LOG="$ROOT/logs/v8_msf_8811.log"
PID_FILE="$ROOT/v8_msf_8811.pid"

if [ ! -x "$PY" ]; then
  echo "missing $PY" >&2
  exit 2
fi

"$PY" - <<'PY'
import json
from pathlib import Path
p=Path("models/v8_multiscale_fusion_candidate.json")
if not p.is_file():
    raise SystemExit("missing multi-scale candidate; run bash run_v8_multiscale_fusion.sh first")
d=json.loads(p.read_text(encoding="utf-8"))
if not d.get("offline_gate_pass"):
    raise SystemExit("multi-scale offline gate did not pass")
print("enabled:", d.get("enabled_targets"))
print("flat RMSE ratio:", d.get("global",{}).get("flat_rmse_ratio"))
print("proxy gain:", d.get("global",{}).get("global_proxy_gain"))
print("trend gain:", d.get("global",{}).get("global_trend_gain"))
PY

oldpid="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print}' | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1)"
if [ -n "${oldpid:-}" ]; then
  kill "$oldpid" 2>/dev/null || true
  sleep 1
  kill -9 "$oldpid" 2>/dev/null || true
fi

mkdir -p "$ROOT/logs"

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
  --host 127.0.0.1 \
  --port "$PORT" \
  --workers 1 \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "V8-MSF pid=$PID"

"$PY" - "$PORT" <<'PY'
import json,sys,time,urllib.request
port=int(sys.argv[1])
last=None
for _ in range(120):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
            h=json.load(r)
        print("health:", h)
        if (
            h.get("status")=="ok"
            and h.get("version")=="2.9.0-v8-msf"
            and h.get("ensemble_version")==8
            and h.get("multiscale_fusion") is True
            and h.get("model_loaded") is True
        ):
            raise SystemExit(0)
        last=h
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"V8-MSF health timeout: {last}")
PY

BASE_PORT=8800 \
API_PORT=8811 \
CALLBACK_PORT=8898 \
N_REQUESTS=50 \
WAIT_TIMEOUT=600 \
"$PY" test_v8_msf_online.py
