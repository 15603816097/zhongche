#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv-full"
PY="$VENV/bin/python"

REQUIRED_MODELS=(
  model_lgb.pkl
  scaler.pkl
  model_xgb.pkl
  scaler_xgb.pkl
  ensemble_config.pkl
  model_pca_xgb.pkl
  preprocess_pca_xgb.pkl
)

echo "============================================================"
echo "FULL ARCHITECTURE - STAGE 1 BOOTSTRAP"
echo "============================================================"

echo
echo "[1/5] Create isolated environment (reuse working CUDA torch)..."
if [ ! -x "$PY" ]; then
  python -m venv --system-site-packages "$VENV"
fi

"$PY" -m pip install -q --upgrade pip setuptools wheel
"$PY" -m pip install -q -r requirements.txt requests scipy joblib

echo
echo "[2/5] Verify Python/CUDA stack..."
"$PY" - <<'PY'
import sys
import numpy, pandas, sklearn, lightgbm, xgboost, fastapi, uvicorn, torch
print("python      :", sys.executable)
print("numpy       :", numpy.__version__)
print("pandas      :", pandas.__version__)
print("sklearn     :", sklearn.__version__)
print("lightgbm    :", lightgbm.__version__)
print("xgboost     :", xgboost.__version__)
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside .venv-full")
print("gpu         :", torch.cuda.get_device_name(0))
print("capability  :", torch.cuda.get_device_capability(0))
PY

echo
echo "[3/5] Find an existing complete V8 model pack..."
mkdir -p models

is_complete_pack() {
  local d="$1"
  [ -d "$d" ] || return 1
  local f
  for f in "${REQUIRED_MODELS[@]}"; do
    [ -f "$d/$f" ] || return 1
  done
  return 0
}

FOUND=""
CANDIDATES=(
  "/root/rail_forecast_v8_deploy/models"
  "/root/v9_lab/models"
  "/root/zhongche/models"
  "/root/rail_forecast/models"
)

for d in "${CANDIDATES[@]}"; do
  if is_complete_pack "$d"; then
    FOUND="$d"
    break
  fi
done

if [ -z "$FOUND" ]; then
  while IFS= read -r cfg; do
    d="$(dirname "$cfg")"
    if [ "$d" != "$ROOT/models" ] && is_complete_pack "$d"; then
      FOUND="$d"
      break
    fi
  done < <(find /root -type f -name ensemble_config.pkl 2>/dev/null | head -n 50)
fi

if [ -n "$FOUND" ]; then
  echo "complete V8 pack found: $FOUND"
  for f in "${REQUIRED_MODELS[@]}"; do
    if [ ! -e "models/$f" ]; then
      ln -s "$FOUND/$f" "models/$f"
    fi
  done
else
  echo "No complete V8 model pack found on this server."
fi

echo
echo "[4/5] Inspect V8 configuration if available..."
if is_complete_pack "$ROOT/models"; then
  "$PY" - <<'PY'
import pickle
from pathlib import Path
p = Path("models/ensemble_config.pkl")
with p.open("rb") as f:
    c = pickle.load(f)
print("version          :", c.get("version"))
print("trajectory_model :", c.get("trajectory_model"))
print("validation_rmse  :", c.get("validation_rmse"))
print("direction_acc    :", c.get("validation_direction_accuracy"))
if int(c.get("version", -1)) != 8:
    raise SystemExit("ERROR: model pack is not V8")
if not str(c.get("trajectory_model", "")).startswith("pca_xgb"):
    raise SystemExit("ERROR: model pack trajectory_model is not V8 PCA")
PY
else
  echo "V8 config check skipped: model pack is still missing."
fi

echo
echo "[5/5] Final state..."
echo "environment: READY"
if is_complete_pack "$ROOT/models"; then
  echo "V8 model pack: READY"
  ls -lh models/model_lgb.pkl models/model_xgb.pkl models/model_pca_xgb.pkl models/ensemble_config.pkl
  echo
  echo "STAGE 1 PASS"
  exit 0
fi

echo "V8 model pack: MISSING"
echo
echo "Upload/copy these seven exact V8 files into:"
echo "  $ROOT/models/"
printf '  %s\n' "${REQUIRED_MODELS[@]}"
echo
echo "Then rerun: bash run_full_arch_stage1_bootstrap.sh"
exit 2
