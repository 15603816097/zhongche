#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "======================================================================"
echo "FULL ARCHITECTURE V1 - INSTALL"
echo "======================================================================"

if [ ! -f requirements.runtime.lock.txt ]; then
  echo "missing requirements.runtime.lock.txt" >&2
  exit 2
fi
if [ ! -f SHA256SUMS ]; then
  echo "missing SHA256SUMS" >&2
  exit 2
fi

echo "[1/5] Verify bundle integrity..."
sha256sum -c SHA256SUMS

PY=""
if command -v python3.12 >/dev/null 2>&1; then
  PY="$(command -v python3.12)"
fi

if [ -n "$PY" ]; then
  echo "[2/5] Create Python 3.12 virtual environment..."
  rm -rf .runtime
  "$PY" -m venv .runtime
  RPY="$ROOT/.runtime/bin/python"
else
  echo "[2/5] Python 3.12 not found; bootstrap local Miniconda..."
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|amd64) CONDA_ARCH="x86_64" ;;
    aarch64|arm64) CONDA_ARCH="aarch64" ;;
    *)
      echo "unsupported architecture: $ARCH" >&2
      exit 2
      ;;
  esac

  MINICONDA="$ROOT/.miniconda"
  INSTALLER="/tmp/miniconda-full-arch-$$.sh"
  URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${CONDA_ARCH}.sh"

  rm -rf "$MINICONDA"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$URL" -o "$INSTALLER"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$INSTALLER" "$URL"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$URL" "$INSTALLER" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  else
    echo "need curl, wget, or python3 to download Miniconda" >&2
    exit 2
  fi

  bash "$INSTALLER" -b -p "$MINICONDA"
  rm -f "$INSTALLER"
  "$MINICONDA/bin/conda" create -y -p "$ROOT/.runtime" python=3.12.7 pip
  RPY="$ROOT/.runtime/bin/python"
fi

echo "[3/5] Install pinned runtime dependencies..."
"$RPY" -m pip install --upgrade pip setuptools wheel
"$RPY" -m pip install -r requirements.runtime.lock.txt

echo "[4/5] Verify frozen runtime prediction..."
PREDICT_WORKERS=2 LGB_INFER_THREADS=8 "$RPY" verify_install.py

echo "[5/5] Installation complete."
echo
echo "Start:"
echo "  bash start.sh"
echo
echo "Health:"
echo "  bash health.sh"
echo
echo "Predict endpoint:"
echo "  http://<server-ip>:8800/predict"
