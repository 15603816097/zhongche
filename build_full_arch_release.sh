#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="$ROOT/.venv-full/bin/python"
OUT_ROOT="/root/full_arch_release"
NAME="full_arch_v1_official_52.895_20260918"
STAGE="$OUT_ROOT/$NAME"
ARCHIVE="$OUT_ROOT/$NAME.tar.gz"
ARCHIVE_SHA="$ARCHIVE.sha256"

MODELS=(
  model_lgb.pkl
  scaler.pkl
  model_xgb.pkl
  scaler_xgb.pkl
  ensemble_config.pkl
  model_pca_xgb.pkl
  preprocess_pca_xgb.pkl
)

if [ ! -x "$PY" ]; then
  echo "missing .venv-full; build must run on the validated server" >&2
  exit 2
fi

for m in "${MODELS[@]}"; do
  if [ ! -f "$ROOT/models/$m" ]; then
    echo "missing model: models/$m" >&2
    exit 2
  fi
done

for i in 1 2 3 4 5; do
  seq="$(printf 'sequence%04d' "$i")"
  if [ ! -f "$ROOT/data/raw/$seq/history.csv" ]; then
    echo "missing official history: data/raw/$seq/history.csv" >&2
    exit 2
  fi
done

if [ ! -f "$ROOT/full_arch_frozen_gate_v1.json" ]; then
  echo "missing frozen gate config" >&2
  exit 2
fi
if [ ! -f "$ROOT/artifacts/full_arch/freeze/full_arch_candidate_v1_manifest.json" ]; then
  echo "missing Stage 8 freeze manifest" >&2
  exit 2
fi
if [ ! -f "$ROOT/artifacts/full_arch/dynamic_gate/stage5_predictions.npz" ]; then
  echo "missing Stage 5 frozen predictions" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
rm -rf "$STAGE"
rm -f "$ARCHIVE" "$ARCHIVE_SHA"
mkdir -p "$STAGE"

echo "======================================================================"
echo "BUILD FULL ARCHITECTURE V1 PORTABLE RELEASE"
echo "======================================================================"
echo "source : $ROOT"
echo "output : $ARCHIVE"
echo

echo "[1/8] Export tracked source tree..."
git archive --format=tar HEAD | tar -xf - -C "$STAGE"

echo "[2/8] Copy exact official model pack..."
mkdir -p "$STAGE/models"
for m in "${MODELS[@]}"; do
  cp -a "$ROOT/models/$m" "$STAGE/models/$m"
done

echo "[3/8] Copy only runtime analog histories..."
rm -rf "$STAGE/data/raw"
for i in 1 2 3 4 5; do
  seq="$(printf 'sequence%04d' "$i")"
  mkdir -p "$STAGE/data/raw/$seq"
  cp -a "$ROOT/data/raw/$seq/history.csv" "$STAGE/data/raw/$seq/history.csv"
done

echo "[4/8] Copy freeze evidence and expected prediction..."
mkdir -p "$STAGE/validation"
cp -a   "$ROOT/artifacts/full_arch/freeze/full_arch_candidate_v1_manifest.json"   "$STAGE/validation/full_arch_candidate_v1_manifest.json"
cp -a   "$ROOT/artifacts/full_arch/freeze/full_arch_candidate_v1_summary.txt"   "$STAGE/validation/full_arch_candidate_v1_summary.txt"

"$PY" - "$ROOT/artifacts/full_arch/dynamic_gate/stage5_predictions.npz" "$STAGE/validation/sequence0001_expected.npy" <<'PY'
import sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
with np.load(src) as z:
    arr = np.asarray(z["final_known_prediction"][0], dtype=np.float64)
np.save(dst, arr)
print("saved expected prediction:", arr.shape)
PY

echo "[5/8] Build pinned runtime dependency lock from validated environment..."
"$PY" - "$STAGE/requirements.runtime.lock.txt" <<'PY'
from importlib.metadata import version, PackageNotFoundError
import sys

packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "fastapi",
    "uvicorn",
    "requests",
    "scipy",
    "joblib",
    "pydantic",
    "starlette",
]
lines = []
for name in packages:
    try:
        v = version(name)
    except PackageNotFoundError:
        raise SystemExit(f"validated environment missing required package: {name}")
    lines.append(f"{name}=={v}")
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[6/8] Install portable launch helpers..."
cp "$ROOT/deploy/full_arch_v1/install.sh" "$STAGE/install.sh"
cp "$ROOT/deploy/full_arch_v1/start.sh" "$STAGE/start.sh"
cp "$ROOT/deploy/full_arch_v1/stop.sh" "$STAGE/stop.sh"
cp "$ROOT/deploy/full_arch_v1/health.sh" "$STAGE/health.sh"
cp "$ROOT/deploy/full_arch_v1/verify_install.py" "$STAGE/verify_install.py"
chmod +x "$STAGE/install.sh" "$STAGE/start.sh" "$STAGE/stop.sh" "$STAGE/health.sh"

cat > "$STAGE/DEPLOY_README.txt" <<'EOF'
FULL ARCHITECTURE V1 - OFFICIAL BEST PACKAGE
Official weighted score: 52.895
Accuracy: 52.01
Trend consistency: 38.84
Robustness: 46.08
Runtime: 72.10
Compliance: 100.00

Fresh server:
  tar -xzf full_arch_v1_official_52.895_20260918.tar.gz
  cd full_arch_v1_official_52.895_20260918
  bash install.sh
  bash start.sh

Health:
  bash health.sh

Prediction endpoint:
  POST http://<server-ip>:8800/predict

Stop:
  bash stop.sh

Notes:
- Linux x86_64 / aarch64.
- First install needs Internet access for Python packages.
- Python 3.12 is used; install.sh bootstraps local Miniconda when needed.
- No GPU is required for inference.
- TCN/PatchTST checkpoints are intentionally not needed because their direct
  forecast weights are zero in the official candidate.
EOF

echo "[7/8] Create integrity checksums..."
(
  cd "$STAGE"
  {
    for m in "${MODELS[@]}"; do
      sha256sum "models/$m"
    done
    sha256sum       full_arch_frozen_gate_v1.json       app.py       app_full_arch.py       config.py       src/inference.py       src/v8_runtime.py       src/full_arch_runtime.py       full_arch_stage4_confidence_ood.py       v9_analog_multiscale_diagnostic.py       validation/sequence0001_expected.npy       requirements.runtime.lock.txt
    for i in 1 2 3 4 5; do
      seq="$(printf 'sequence%04d' "$i")"
      sha256sum "data/raw/$seq/history.csv"
    done
  } > SHA256SUMS
)

echo "[8/8] Create tar.gz..."
tar -C "$OUT_ROOT" -cf - "$NAME" | gzip -1 > "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE_SHA"

echo
echo "======================================================================"
echo "PACKAGE COMPLETE"
echo "======================================================================"
ls -lh "$ARCHIVE" "$ARCHIVE_SHA"
echo
cat "$ARCHIVE_SHA"
echo
echo "Download from your local computer with:"
echo "  scp -P 24440 root@180.127.11.177:$ARCHIVE ~/Downloads/"
