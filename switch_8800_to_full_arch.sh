#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="$ROOT/.venv-full/bin/python"
PORT=8800
PUBLIC_HOST="180.127.11.177"
PUBLIC_PORT="24676"

PROD_DIR="$ROOT/artifacts/full_arch/production"
PID_FILE="$PROD_DIR/full_arch_8800.pid"
LOG_FILE="$PROD_DIR/full_arch_8800.log"
BEFORE_HEALTH="$PROD_DIR/health_before_cutover.json"
AFTER_HEALTH="$PROD_DIR/health_after_cutover.json"

FROZEN_CONFIG="$ROOT/full_arch_frozen_gate_v1.json"
FREEZE_MANIFEST="$ROOT/artifacts/full_arch/freeze/full_arch_candidate_v1_manifest.json"
STAGE5_NPZ="$ROOT/artifacts/full_arch/dynamic_gate/stage5_predictions.npz"

mkdir -p "$PROD_DIR"

if [ ! -x "$PY" ]; then
  echo "missing $PY" >&2
  exit 2
fi
for f in "$FROZEN_CONFIG" "$FREEZE_MANIFEST" "$STAGE5_NPZ"; do
  if [ ! -f "$f" ]; then
    echo "missing required frozen artifact: $f" >&2
    exit 2
  fi
done

listener_pid() {
  ss -ltnp 2>/dev/null     | awk -v p=":$PORT" '$4 ~ p"$" {print}'     | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p'     | head -n 1
}

wait_port_free() {
  "$PY" - <<'PY'
import socket, time
for _ in range(80):
    s = socket.socket()
    s.settimeout(0.2)
    try:
        rc = s.connect_ex(("127.0.0.1", 8800))
    finally:
        s.close()
    if rc != 0:
        raise SystemExit(0)
    time.sleep(0.25)
raise SystemExit("port 8800 is still occupied")
PY
}

wait_health_version() {
  local expected="$1"
  "$PY" - "$expected" <<'PY'
import json, sys, time
import requests
expected = sys.argv[1]
last = None
for _ in range(180):
    try:
        r = requests.get("http://127.0.0.1:8800/health", timeout=2)
        h = r.json()
        print("HEALTH:", json.dumps(h, ensure_ascii=False))
        if (
            r.status_code == 200
            and h.get("status") == "ok"
            and h.get("version") == expected
            and h.get("ensemble_version") == 8
            and h.get("model_loaded") is True
        ):
            raise SystemExit(0)
        last = f"unexpected health={h}"
    except SystemExit:
        raise
    except Exception as exc:
        last = repr(exc)
    time.sleep(0.5)
raise SystemExit(f"health timeout for version={expected}: {last}")
PY
}

start_v8_fallback() {
  echo "[ROLLBACK] Starting exact V8 fallback from full_arch_lab..."
  cd "$ROOT"
  PREDICT_WORKERS=2   LGB_INFER_THREADS=8   CALLBACK_TIMEOUT=20   CALLBACK_RETRIES=5   CALLBACK_MIN_AGE=1.0   CALLBACK_GAP=0.25   PYTHONUNBUFFERED=1   nohup "$PY" -m uvicorn app:app     --host 0.0.0.0     --port "$PORT"     --workers 1     > "$PROD_DIR/v8_rollback_8800.log" 2>&1 &
  local rpid=$!
  echo "$rpid" > "$PROD_DIR/v8_rollback_8800.pid"
  if ! wait_health_version "2.9.0"; then
    echo "[ROLLBACK FAILED] V8 did not become healthy." >&2
    tail -n 100 "$PROD_DIR/v8_rollback_8800.log" >&2 || true
    return 1
  fi
  echo "[ROLLBACK PASS] Exact V8 is serving on 8800."
}

rollback_on_failure() {
  local rc=$?
  echo
  echo "[CUTOVER FAILED] rc=$rc. Attempting automatic V8 rollback..." >&2
  local pid
  pid="$(listener_pid || true)"
  if [ -n "${pid:-}" ]; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  wait_port_free || true
  start_v8_fallback || true
  exit "$rc"
}

echo "======================================================================"
echo "FULL ARCHITECTURE V1 - PRODUCTION CUTOVER"
echo "======================================================================"
echo "internal endpoint : 0.0.0.0:$PORT"
echo "public endpoint   : http://$PUBLIC_HOST:$PUBLIC_PORT"
echo "frozen candidate  : full_arch_dynamic_gate_v1"
echo "rollback target   : exact V8 (app:app)"
echo

echo "[1/7] Validate frozen candidate metadata..."
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path(".")
cfg = json.loads((root / "full_arch_frozen_gate_v1.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "artifacts/full_arch/freeze/full_arch_candidate_v1_manifest.json").read_text(encoding="utf-8"))
assert cfg.get("frozen") is True
assert cfg.get("global_gate_pass") is True
assert cfg.get("enabled_targets") == ["speed_rpm", "acoustic_db", "pressure_kpa"]
assert manifest.get("candidate") == "full_arch_dynamic_gate_v1"
assert manifest.get("validation", {}).get("stage6_direct_runtime") == "PASS"
assert manifest.get("validation", {}).get("stage7_callback_stress") == "PASS"
print("frozen metadata: PASS")
print("git_head:", manifest.get("git_head"))
print("enabled_targets:", cfg.get("enabled_targets"))
PY

echo
echo "[2/7] Inspect current 8800 health..."
"$PY" - "$BEFORE_HEALTH" <<'PY'
import json, sys
import requests
path = sys.argv[1]
try:
    r = requests.get("http://127.0.0.1:8800/health", timeout=3)
    data = r.json()
except Exception as exc:
    data = {"status": "unreachable", "error": repr(exc)}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(json.dumps(data, ensure_ascii=False, indent=2))
PY

OLD_PID="$(listener_pid || true)"
if [ -n "${OLD_PID:-}" ]; then
  echo "current 8800 listener pid=$OLD_PID"
else
  echo "no current 8800 listener detected"
fi

echo
echo "[3/7] Stop current 8800 listener..."
if [ -n "${OLD_PID:-}" ]; then
  kill "$OLD_PID"
  for _ in $(seq 1 60); do
    kill -0 "$OLD_PID" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "current listener did not exit gracefully; sending SIGKILL"
    kill -9 "$OLD_PID" || true
  fi
fi
wait_port_free
echo "port 8800 is free"

trap rollback_on_failure ERR

echo
echo "[4/7] Start Full Architecture V1 on 8800..."
PREDICT_WORKERS=2 LGB_INFER_THREADS=8 CALLBACK_TIMEOUT=20 CALLBACK_RETRIES=5 CALLBACK_MIN_AGE=1.0 CALLBACK_GAP=0.25 PYTHONUNBUFFERED=1 nohup "$PY" -m uvicorn app_full_arch:app   --host 0.0.0.0   --port "$PORT"   --workers 1   > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "full-arch pid=$NEW_PID"

echo
echo "[5/7] Wait for production health..."
wait_health_version "3.0.0-full-arch"
"$PY" - "$AFTER_HEALTH" <<'PY'
import json, sys, requests
r = requests.get("http://127.0.0.1:8800/health", timeout=5)
data = r.json()
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(json.dumps(data, ensure_ascii=False, indent=2))
PY

echo
echo "[6/7] Run frozen sequence reproduction probe through production API..."
"$PY" - <<'PY'
import numpy as np
import pandas as pd
import requests

from config import DATA_DIR, HORIZON, TARGET_COLUMNS

with np.load("artifacts/full_arch/dynamic_gate/stage5_predictions.npz") as z:
    expected = np.asarray(z["final_known_prediction"][0], dtype=np.float64)

hdf = pd.read_csv(DATA_DIR / "sequence0001" / "history.csv")

def safe(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None

history = [
    {
        "step": int(i),
        "values": {c: safe(row[c]) for c in TARGET_COLUMNS},
    }
    for i, row in hdf.iterrows()
]
payload = {
    "requestId": "FULL_ARCH_PROD_PROBE_SEQ0001",
    "history_length": len(history),
    "forecast_horizon": HORIZON,
    "target_columns": list(TARGET_COLUMNS),
    "history": history,
}
r = requests.post("http://127.0.0.1:8800/predict", json=payload, timeout=120)
r.raise_for_status()
data = r.json()
if data.get("code") != 0 or len(data.get("predictions", [])) != HORIZON:
    raise SystemExit(f"bad production probe response: {data}")
pred = np.asarray(
    [[row["values"][c] for c in TARGET_COLUMNS] for row in data["predictions"]],
    dtype=np.float64,
)
diff = float(np.max(np.abs(pred - expected)))
print(f"production probe max_abs_diff={diff:.3e}")
if diff > 2e-5:
    raise SystemExit(f"production reproduction mismatch: {diff}")
print("production frozen reproduction: PASS")
PY

echo
echo "[7/7] Finalize cutover..."
trap - ERR

echo "LOCAL CUTOVER: PASS"
echo "full-arch is serving on internal port 8800"
echo "public mapped endpoint: http://$PUBLIC_HOST:$PUBLIC_PORT"
echo "public health endpoint: http://$PUBLIC_HOST:$PUBLIC_PORT/health"
echo "public predict endpoint: http://$PUBLIC_HOST:$PUBLIC_PORT/predict"
echo "log: $LOG_FILE"
echo
echo "Manual rollback command:"
echo "  cd $ROOT && bash rollback_8800_to_v8_full_arch.sh"
echo

# Best-effort public mapping check. Do not fail cutover if hairpin NAT blocks it.
"$PY" - "$PUBLIC_HOST" "$PUBLIC_PORT" <<'PY'
import json, sys, requests
host, port = sys.argv[1], sys.argv[2]
url = f"http://{host}:{port}/health"
try:
    r = requests.get(url, timeout=5)
    print("PUBLIC HEALTH:", r.status_code, json.dumps(r.json(), ensure_ascii=False))
except Exception as exc:
    print("PUBLIC HEALTH CHECK WARNING:", repr(exc))
    print("Local 8800 is healthy; external hairpin/NAT may block self-check.")
PY
