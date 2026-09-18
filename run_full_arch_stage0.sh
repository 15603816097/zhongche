#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "FULL ARCHITECTURE - STAGE 0 ENVIRONMENT AUDIT"
echo "============================================================"

echo
echo "[Git]"
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short

echo
echo "[System]"
uname -a
df -h .
free -h || true

echo
echo "[GPU]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi: NOT FOUND"
fi

echo
echo "[Python]"
PY="${FULL_ARCH_PYTHON:-python}"
"$PY" --version
"$PY" - <<'PY'
import sys
print("executable:", sys.executable)
mods = ["numpy","pandas","sklearn","lightgbm","xgboost","torch","fastapi","uvicorn"]
for name in mods:
    try:
        m = __import__(name)
        print(f"{name:10s}: OK  {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"{name:10s}: MISSING/ERROR  {e!r}")
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.version.cuda      :", torch.version.cuda)
    if torch.cuda.is_available():
        print("cuda device            :", torch.cuda.get_device_name(0))
        print("cuda capability        :", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch cuda audit error:", repr(e))
PY

echo
echo "[Official data]"
for s in sequence0001 sequence0002 sequence0003 sequence0004 sequence0005; do
  hp="data/raw/$s/history.csv"
  fp="data/raw/$s/future.csv"
  if [ -f "$hp" ] && [ -f "$fp" ]; then
    echo "$s: OK"
  else
    echo "$s: MISSING history/future"
  fi
done

echo
echo "[V8 model pack]"
for f in \
  models/model_lgb.pkl \
  models/scaler.pkl \
  models/model_xgb.pkl \
  models/scaler_xgb.pkl \
  models/ensemble_config.pkl \
  models/model_pca_xgb.pkl \
  models/preprocess_pca_xgb.pkl
do
  if [ -e "$f" ]; then
    ls -lh "$f"
  else
    echo "MISSING: $f"
  fi
done

echo
echo "[Existing deep artifacts]"
find models -maxdepth 1 -type f \( -iname '*patchtst*' -o -iname '*tcn*' -o -iname '*deep*' \) -printf '%f %s bytes\n' 2>/dev/null | sort || true

echo
echo "[Key source hashes]"
sha256sum \
  src/inference.py \
  src/v8_runtime.py \
  v9_analog_multiscale_diagnostic.py \
  src/deep/patchtst_forecaster.py \
  src/deep/tcn_forecaster.py 2>/dev/null || true

echo
echo "============================================================"
echo "STAGE 0 COMPLETE"
echo "Do not modify V8 files yet."
echo "============================================================"
