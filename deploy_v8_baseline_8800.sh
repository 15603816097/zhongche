#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ARCHIVE="${1:-/root/full_arch_v1_official_52.895_20260918.tar.gz}"
BUNDLE_NAME="full_arch_v1_official_52.895_20260918"
PORT="${PORT:-8800}"
PUBLIC_HOST="${PUBLIC_HOST:-180.127.11.177}"
PUBLIC_PORT="${PUBLIC_PORT:-24188}"

VENV="$ROOT/.venv-v8"
PY="$VENV/bin/python"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/v8_baseline_8800.log"
PID_FILE="$ROOT/v8_baseline_8800.pid"

MODELS=(
  model_lgb.pkl
  scaler.pkl
  model_xgb.pkl
  scaler_xgb.pkl
  ensemble_config.pkl
  model_pca_xgb.pkl
  preprocess_pca_xgb.pkl
)

listener_pid() {
  ss -ltnp 2>/dev/null \
    | awk -v p=":$PORT" '$4 ~ p"$" {print}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -n 1
}

wait_port_free() {
  local pybin="${1:-python3}"
  "$pybin" - "$PORT" <<'PY'
import socket, sys, time
port=int(sys.argv[1])
for _ in range(80):
    s=socket.socket(); s.settimeout(0.2)
    try:
        rc=s.connect_ex(("127.0.0.1",port))
    finally:
        s.close()
    if rc != 0:
        raise SystemExit(0)
    time.sleep(0.25)
raise SystemExit(f"port {port} is still occupied")
PY
}

echo "======================================================================"
echo "V8 BASELINE DEPLOYMENT"
echo "======================================================================"
echo "repo            : $ROOT"
echo "internal port   : $PORT"
echo "public endpoint : http://$PUBLIC_HOST:$PUBLIC_PORT/predict"
echo

echo "[1/7] Prepare exact V8 model pack..."
mkdir -p "$ROOT/models"
missing=0
for m in "${MODELS[@]}"; do
  [ -f "$ROOT/models/$m" ] || missing=1
done

if [ "$missing" -eq 1 ]; then
  if [ ! -f "$ARCHIVE" ]; then
    echo "V8 model files are missing and release archive was not found:" >&2
    echo "  $ARCHIVE" >&2
    echo "Upload the archive first, or pass its path as argument." >&2
    exit 2
  fi

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "Extracting model pack from: $ARCHIVE"
  tar -xzf "$ARCHIVE" -C "$TMP" "$BUNDLE_NAME/models"
  for m in "${MODELS[@]}"; do
    cp -a "$TMP/$BUNDLE_NAME/models/$m" "$ROOT/models/$m"
  done
  rm -rf "$TMP"
  trap - EXIT
fi

for m in "${MODELS[@]}"; do
  if [ ! -f "$ROOT/models/$m" ]; then
    echo "missing required model: models/$m" >&2
    exit 2
  fi
done
echo "model pack: READY"

echo
echo "[2/7] Prepare isolated Python 3.12 environment..."
if [ ! -x "$PY" ]; then
  if command -v python3.12 >/dev/null 2>&1; then
    if ! python3.12 -m venv "$VENV"; then
      rm -rf "$VENV"
    fi
  fi
fi

if [ ! -x "$PY" ]; then
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|amd64) CONDA_ARCH="x86_64" ;;
    aarch64|arm64) CONDA_ARCH="aarch64" ;;
    *) echo "unsupported architecture: $ARCH" >&2; exit 2 ;;
  esac

  MINICONDA="$ROOT/.miniconda-v8"
  INSTALLER="/tmp/miniconda-v8-$.sh"
  URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${CONDA_ARCH}.sh"

  if [ ! -x "$MINICONDA/bin/conda" ]; then
    if command -v curl >/dev/null 2>&1; then
      curl -fL "$URL" -o "$INSTALLER"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$INSTALLER" "$URL"
    else
      echo "curl or wget is required to bootstrap Miniconda." >&2
      exit 2
    fi
    bash "$INSTALLER" -b -p "$MINICONDA"
    rm -f "$INSTALLER"
  fi
  "$MINICONDA/bin/conda" create -y -p "$VENV" python=3.12.7 pip
fi

"$PY" -m pip install -q --upgrade pip setuptools wheel
"$PY" -m pip install -q -r requirements.v8.runtime.lock.txt
echo "python: $("$PY" --version)"
echo "runtime env: READY"

echo
echo "[3/7] Verify active ensemble is exact V8..."
"$PY" - <<'PY'
import pickle
from pathlib import Path
p=Path("models/ensemble_config.pkl")
with p.open("rb") as f:
    c=pickle.load(f)
print("version:", c.get("version"))
print("trajectory_model:", c.get("trajectory_model"))
if int(c.get("version",-1)) != 8:
    raise SystemExit("ensemble_config is not V8")
if str(c.get("trajectory_model","")) != "pca_xgb_source_aware_hf_v1":
    raise SystemExit("unexpected trajectory_model")
print("V8 config: PASS")
PY

echo
echo "[4/7] Stop any current listener on $PORT..."
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
wait_port_free "$PY"
echo "port $PORT: FREE"

echo
echo "[5/7] Start exact V8 on $PORT..."
mkdir -p "$LOG_DIR"
PREDICT_WORKERS=2 \
LGB_INFER_THREADS=8 \
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

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "pid=$NEW_PID"

echo
echo "[6/7] Wait for V8 health..."
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
            and h.get("version")=="2.9.0"
            and h.get("ensemble_version")==8
            and h.get("model_loaded") is True
        ):
            print("V8 HEALTH: PASS")
            raise SystemExit(0)
        last=h
    except SystemExit:
        raise
    except Exception as exc:
        last=repr(exc)
    time.sleep(0.5)
raise SystemExit(f"V8 health timeout: {last}")
PY

echo
echo "[7/7] Final deployment info..."
echo "V8 BASELINE DEPLOY: PASS"
echo "internal health : http://127.0.0.1:$PORT/health"
echo "public health   : http://$PUBLIC_HOST:$PUBLIC_PORT/health"
echo "official URL    : http://$PUBLIC_HOST:$PUBLIC_PORT/predict"
echo "log             : $LOG_FILE"
